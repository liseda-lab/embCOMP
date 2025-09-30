# =============================
# Verbose Debug + Timestamped Logs with Embedding Caching, Validation, Checkpointing, CSV Logging, and Early Stopping
# =============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import os
from datetime import datetime

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def save_checkpoint(classifier, optimizer, scheduler, epoch, train_losses, checkpoint_path):
    checkpoint = {
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
    }
    torch.save(checkpoint, checkpoint_path)
    log(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(classifier, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    log(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch+1}")
    return epoch, train_losses

def append_metrics_to_csv(metrics, csv_path, info_dict):
    row = {**info_dict, **metrics}
    df = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def experiment_already_done(metrics_csv, lr, loss_name, batch_size):
    if not os.path.exists(metrics_csv):
        return False
    df = pd.read_csv(metrics_csv)
    done = (
        (df['lr'] == lr) &
        (df['loss_function'] == loss_name) &
        (df['batch_size'] == batch_size)
    )
    return done.any()

# =============================
# Dataset Class & Utilities
# =============================
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def preprocess_sequence(seq):
    spaced = " ".join(seq.upper())
    return f"[CLS] {spaced} [SEP]"

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    sequences = df["Sequence"].apply(preprocess_sequence).tolist()
    labels = df["Viability"].tolist()

    log(f"Loaded {len(sequences)} sequences from {csv_path}")
    log("First 5 preprocessed sequences:")
    for seq in sequences[:5]:
        log(f" - {seq[:100]}...")

    return sequences, labels

def get_dataloader(sequences, labels, batch_size=1, shuffle=True):
    dataset = SequenceDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# =============================
# Classifier Model
# =============================
class PBLinearClassifier(nn.Module):
    def __init__(self, model, tokenizer, device):
        super(PBLinearClassifier, self).__init__()
        self.model = model
        self.classify = nn.Linear(self.model.config.hidden_size, 2).to(device)
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).to(self.device)#pooler_output.to(self.device)
        logits = self.classify(embedding).to(self.device)
        return logits

    def get_tokens(self, sequences):
        tokens = self.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        return {k: v for k, v in tokens.items() if k in ['input_ids', 'attention_mask']}

    def train_model(self, train_dataloader, optimizer, criterion):
        self.model.train()
        self.classify.train()
        log("Set model to training mode...")
        loss_list = []
        sequence_counter = 0

        for i, (batched_sequences, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            tokens = self.get_tokens(batched_sequences)
            outputs = self(**tokens)

            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(self.device)

            one_hot_labels = torch.eye(2, device=self.device)[labels]
            loss = criterion(torch.softmax(outputs, dim=1), one_hot_labels)
            loss = 1 - torch.mean(loss) if isinstance(criterion, CosineLoss) else loss
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            sequence_counter += len(batched_sequences)

            if sequence_counter % 100 == 0:
                log(f"Forwarded {sequence_counter} sequences so far...")

        average_loss = sum(loss_list) / len(train_dataloader)
        return average_loss, loss_list

    def extract_embeddings(self, dataloader, save_path):
        if os.path.exists(save_path):
            log(f"Embeddings already exist at {save_path}, loading...")
            data = torch.load(save_path)
            embeddings_tensor = data['embeddings']
            labels_tensor = data['labels']
            sequences = data['sequences']
        else:
            log(f"Embeddings not found at {save_path}, generating now...")
            self.model.eval()
            embeddings = []
            labels = []
            sequences = []
            sequence_count = 0

            with torch.no_grad():
                for batched_sequences, batch_labels in dataloader:
                    tokens = self.get_tokens(batched_sequences)
                    outputs = self.model(**tokens)
                    embedding = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).cpu()
                    embeddings.append(embedding)
                    labels.extend(batch_labels)
                    sequences.extend(batched_sequences)
                    sequence_count += len(batched_sequences)
                    if sequence_count % 10000 == 0:
                        log(f"Forwarded {sequence_count} sequences so far...")

            embeddings_tensor = torch.cat(embeddings)
            labels_tensor = torch.tensor(labels)
            torch.save({'embeddings': embeddings_tensor, 'labels': labels_tensor, 'sequences': sequences}, save_path)
            log(f"Saved embeddings to {save_path}")

        log("\n--- Example embedding details ---")
        log("First original sequence:")
        log(sequences[0])
        log(f"Embedding shape: {embeddings_tensor[0].shape}")
        log(f"First 10 elements of first embedding: {embeddings_tensor[0][:10].numpy()}")

    def evaluate_model(self, dataloader):
        self.model.eval()
        self.classify.eval()
        preds = []
        true_labels = []

        with torch.no_grad():
            for sequences, labels in dataloader:
                tokens = self.get_tokens(sequences)
                outputs = self(**tokens)
                probs = torch.softmax(outputs, dim=1)
                pred_labels = torch.argmax(probs, dim=1).cpu().numpy()
                preds.extend(pred_labels)
                true_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

        acc = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds)
        recall = recall_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)
        auc = roc_auc_score(true_labels, preds)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc
        }

# =============================
# Train/Test Entry Point
# =============================
train_data = "/home/lferraz/datasets/data_train.csv"
val_data = "/home/lferraz/datasets/data_val.csv"
test_data = "/home/lferraz/datasets/data_test.csv"

if __name__ == "__main__":
    BATCH_SIZE = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    log("Loading data...")

    train_seqs, train_labels = load_dataset(train_data)
    val_seqs, val_labels = load_dataset(val_data)
    test_seqs, test_labels = load_dataset(test_data)

    train_loader = get_dataloader(train_seqs, train_labels, batch_size=BATCH_SIZE)
    val_loader = get_dataloader(val_seqs, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_seqs, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    loss_functions = {
        "bce": nn.BCELoss()
    }
    
    learning_rates = [10**-6, 10**-7]#]np.logspace(-6, -4, num=3)
    epochs = 5
    save_dir = "/home/lferraz/datasets/fine_tuned_models_test_mean"
    os.makedirs(save_dir, exist_ok=True)

    from torch.optim.lr_scheduler import StepLR

    metrics_csv = os.path.join(save_dir, "training_metrics.csv")
    patience = 3  # Early stopping patience

    for loss_name, criterion in loss_functions.items():
        for lr in learning_rates:
            if experiment_already_done(metrics_csv, lr, loss_name, BATCH_SIZE):
                log(f"Skipping already completed experiment: LR={lr:.0e}, Loss={loss_name}, BS={BATCH_SIZE}")
                continue

            log(f"\n=== Training with LR: {lr:.0e}, Loss: {loss_name}, Batch size {BATCH_SIZE} ===")

            model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
            tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
            classifier = PBLinearClassifier(model, tokenizer, device).to(device)

            classifier.extract_embeddings(test_loader, os.path.join(save_dir, f"embeddings_no_training.pt"))

            optimizer = optim.Adam(classifier.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=3, gamma=0.05)

            checkpoint_path = os.path.join(save_dir, f"checkpoint_lr{lr:.0e}_loss{loss_name}_bs{BATCH_SIZE}.pth")
            start_epoch = 0
            train_losses = []

            if os.path.exists(checkpoint_path):
                start_epoch, train_losses = load_checkpoint(classifier, optimizer, scheduler, checkpoint_path)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(start_epoch, epochs):
                log(f"\nStarting epoch {epoch + 1}...")
                avg_loss, _ = classifier.train_model(train_loader, optimizer, criterion)
                scheduler.step()
                train_losses.append(avg_loss)
                log(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

                val_metrics = classifier.evaluate_model(val_loader)
                val_loss = 1 - val_metrics['roc_auc']
                for key, value in val_metrics.items():
                    log(f"Validation {key}: {value:.4f}")

                model_path = os.path.join(save_dir, f"pbclassifier_lr{lr:.0e}_loss{loss_name}_bs{BATCH_SIZE}_{epoch}epochs.pt")
                torch.save(classifier.state_dict(), model_path)
                log(f"Saved model: {model_path}")

                save_checkpoint(classifier, optimizer, scheduler, epoch, train_losses, checkpoint_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        log(f"Early stopping triggered at epoch {epoch+1}!")
                        break

            plt.figure()
            plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
            plt.title(f"Loss Curve (LR: {lr:.0e}, Loss: {loss_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f"loss_plot_lr{lr:.0e}_loss{loss_name}_bs{BATCH_SIZE}.png")
            plt.savefig(plot_path)
            plt.close()
            log(f"Saved loss plot: {plot_path}")

            metrics = classifier.evaluate_model(test_loader)
            log("\nTest set evaluation metrics:")
            for key, value in metrics.items():
                log(f"Test {key}: {value:.4f}")

            info = {"lr": lr, "loss_function": loss_name, "batch_size": BATCH_SIZE}
            append_metrics_to_csv(metrics, metrics_csv, info)

            classifier.extract_embeddings(test_loader, os.path.join(save_dir, f"embeddings_after_lr{lr:.0e}_loss{loss_name}_bs{BATCH_SIZE}_{epochs}epochs.pt"))

            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                log(f"Deleted checkpoint: {checkpoint_path}")
            log(f"Completed training for LR: {lr:.0e}, Loss: {loss_name}, Batch size {BATCH_SIZE}")