#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
from joblib import load, dump
from datetime import datetime
import os

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)


# In[2]:


print('Logistic Regression classifers pre-sampling of data partitions')
print('')


# In[3]:


# Gate for final run
FINAL_RUN = True  # Set to False for testing/dev mode

if FINAL_RUN:
    print('FINAL RUN: Using full dataset and production file paths')
else:
    print('TEST MODE: Sampling dataset and using local/test file paths')


# ### Auxiliary functions

# In[4]:


## Function train_and_evaluate_logR_definedsplit
def train_and_evaluate_logR_definedsplit(train_df, val_df, test_df, model_name, representation):
    print(f"Starting training and evaluation for model: {model_name}")

    # Extract features and target
    X_train = np.vstack(train_df[representation].values)
    y_train = train_df['Viability'].values

    X_val = np.vstack(val_df[representation].values)
    y_val = val_df['Viability'].values

    X_test = np.vstack(test_df[representation].values)
    y_test = test_df['Viability'].values

    # Parameter grid
    param_grid = [
        {'penalty': 'l2', 'C': 0.01},
        {'penalty': 'l2', 'C': 0.1},
        {'penalty': 'l2', 'C': 1},
        {'penalty': 'l2', 'C': 10}
    ]

    # Manual Grid Search on train/val using F1
    best_f1 = -1
    best_params = None

    for params in param_grid:
        model = LogisticRegression(
            tol=0.0001,
            max_iter=10000,
            random_state=42,
            solver='lbfgs',  
            verbose=1,
            **params
        )
        print(f"Training with params: {params}")
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    print(f"\nGrid search completed at: {datetime.now()}")
    print(f"Best Parameters: {best_params}")

    # Train final model on train+val with best params
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])

    final_model = LogisticRegression(
        random_state=42,
        solver='lbfgs',
        **best_params
    )
    final_model.fit(X_trainval, y_trainval)
    print(f"Model training completed at: {datetime.now()}")

    # Save the final model
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    dump(final_model, model_path)

    # Test on holdout set
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    # Compute metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    # Create metrics DataFrame
    cv_metrics = pd.DataFrame({
        'Model': [model_name],
        'Precision': [precision],
        'Accuracy': [accuracy],
        'Recall': [recall],
        'F1 Score': [f1],
        'ROC AUC': [roc_auc]
    })

    print("\nTest Metrics:")
    print(cv_metrics)

    # Save metrics CSV
    metrics_path = os.path.join(save_dir, f'{model_name}_test_metrics.csv')
    cv_metrics.to_csv(metrics_path, index=False)

    # Compute and save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=['True Neg', 'True Pos'], columns=['Pred Neg', 'Pred Pos'])
    cm_path = os.path.join(save_dir, f'{model_name}_test_confusion_matrix.csv')
    cm_df.to_csv(cm_path)

    print(f"\nConfusion matrix saved as '{cm_path}'")
    print(f"Metrics saved as '{metrics_path}'")
    print(f"Model saved as '{model_path}'")
    print(f"Model testing completed at: {datetime.now()}")
    print('')


# In[5]:


def load_split_df(split: str, index: int, final_run: bool, base_path_linux: str, base_path_win: str):
    """Loads a single split (train/val/test) for a given index."""
    filename = f'final_REPs_{split}_{index}.parquet'
    file_path = os.path.join(base_path_linux if final_run else base_path_win, filename)
    df = pd.read_parquet(file_path)

    return df


# In[6]:


## Function 'consolidate_test_metrics'
def consolidate_test_metrics(model_name: str, n_models: int):
    all_metrics = []
    for i in range(1, n_models + 1):
        model_id = f"{model_name}_{i}"
        file_path = os.path.join(save_dir, f"{model_id}_test_metrics.csv")
        df = pd.read_csv(file_path)
        all_metrics.append(df)
        os.remove(file_path)
        print(f"Deleted {file_path}")

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    output_path = os.path.join(save_dir, f"test_metrics_{model_name}_all.csv")
    combined_metrics.to_csv(output_path, index=False)
    print(f"Saved combined metrics to '{output_path}'")

    return combined_metrics


# In[7]:


## Function consolidate_confusion_matrices
def consolidate_confusion_matrices(model_name_prefix: str, n_models: int, save_dir: str, output_excel_name: str):
    excel_file = os.path.join(save_dir, output_excel_name)
    
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        for i in range(1, n_models + 1):
            model_name = f"{model_name_prefix}_{i}"
            file_path = os.path.join(save_dir, f"{model_name}_test_confusion_matrix.csv")
            
            # Read confusion matrix CSV
            cm_df = pd.read_csv(file_path, index_col=0)
            
            # Write to Excel sheet named after the model
            cm_df.to_excel(writer, sheet_name=model_name)
            
            # Delete original CSV
            os.remove(file_path)
            print(f"Deleted {file_path}")

    print(f"Saved all confusion matrices to '{excel_file}'")


# ### OHE

# In[8]:


# Parameters
N_SPLITS = 30  # Number of sets to run
base_path_linux = '/home/afarodrigues/datasets/Filipa'
base_path_win = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation'

# Output directories
save_dir = '/home/afarodrigues/embCOMP/outputs_csv' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'
models_dir = '/home/afarodrigues/embCOMP/outputs_others' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'

print(f"Started job for OHE data at: {datetime.now()}")

# One-at-a-time loop
for i in range(1, N_SPLITS + 1):
    print(f"\n=== Starting job for split {i} at {datetime.now()} ===")

    # Load only split i
    train_df = load_split_df('train', i, FINAL_RUN, base_path_linux, base_path_win)
    val_df   = load_split_df('val', i, FINAL_RUN, base_path_linux, base_path_win)
    test_df  = load_split_df('test', i, FINAL_RUN, base_path_linux, base_path_win)

    # Run training/evaluation
    model_name = f'logR_OHE_model_{i}'
    train_and_evaluate_logR_definedsplit(train_df, val_df, test_df,
                                       model_name=model_name,
                                       representation='OHE-SVD')

    # Memory cleanup 
    del train_df, val_df, test_df

## Run 'consolidate_test_metrics'
combined_df = consolidate_test_metrics("logR_OHE_model", 30)
print('Consolidated metrics for OHE:')
print(combined_df)
print('')

## Run 'consolidate_confusion_matrices'  
consolidate_confusion_matrices(
    model_name_prefix='logR_OHE_model',
    n_models=30,
    save_dir=save_dir,
    output_excel_name='confusion_matrices_logR_OHE_all.xlsx'
)

print(f"Finished job for OHE data at: {datetime.now()}")


# ## cls_raw_EMB

# In[10]:


# Parameters
N_SPLITS = 30  # Number of sets to run
base_path_linux = '/home/afarodrigues/datasets/Filipa'
base_path_win = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation'

# Output directories
save_dir = '/home/afarodrigues/embCOMP/outputs_csv' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'
models_dir = '/home/afarodrigues/embCOMP/outputs_others' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'

print(f"Started job for cls_raw_EMB data at: {datetime.now()}")

# One-at-a-time loop
for i in range(1, N_SPLITS + 1):
    print(f"\n=== Starting job for split {i} at {datetime.now()} ===")

    # Load only split i
    train_df = load_split_df('train', i, FINAL_RUN, base_path_linux, base_path_win)
    val_df   = load_split_df('val', i, FINAL_RUN, base_path_linux, base_path_win)
    test_df  = load_split_df('test', i, FINAL_RUN, base_path_linux, base_path_win)

    # Run training/evaluation
    model_name = f'logR_cls_raw_EMB_model_{i}'
    train_and_evaluate_logR_definedsplit(train_df, val_df, test_df,
                                       model_name=model_name,
                                       representation='cls_raw_EMB')

    # Memory cleanup 
    del train_df, val_df, test_df

## Run 'consolidate_test_metrics'
combined_df = consolidate_test_metrics("logR_cls_raw_EMB_model", 30)
print('Consolidated metrics for cls_raw_EMB:')
print(combined_df)
print('')

## Run 'consolidate_confusion_matrices'  
consolidate_confusion_matrices(
    model_name_prefix='logR_cls_raw_EMB_model',
    n_models=30,
    save_dir=save_dir,
    output_excel_name='confusion_matrices_logR_cls_raw_EMB_all.xlsx'
)

print(f"Finished job for cls_raw_EMB data at: {datetime.now()}")


# ## cls_t_EMB

# In[ ]:


# Parameters
N_SPLITS = 30  # Number of sets to run
base_path_linux = '/home/afarodrigues/datasets/Filipa'
base_path_win = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation'

# Output directories
save_dir = '/home/afarodrigues/embCOMP/outputs_csv' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'
models_dir = '/home/afarodrigues/embCOMP/outputs_others' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'

print(f"Started job for cls_t_EMB data at: {datetime.now()}")

# One-at-a-time loop
for i in range(1, N_SPLITS + 1):
    print(f"\n=== Starting job for split {i} at {datetime.now()} ===")

    # Load only split i
    train_df = load_split_df('train', i, FINAL_RUN, base_path_linux, base_path_win)
    val_df   = load_split_df('val', i, FINAL_RUN, base_path_linux, base_path_win)
    test_df  = load_split_df('test', i, FINAL_RUN, base_path_linux, base_path_win)

    # Run training/evaluation
    model_name = f'logR_cls_t_EMB_model_{i}'
    train_and_evaluate_logR_definedsplit(train_df, val_df, test_df,
                                       model_name=model_name,
                                       representation='cls_t_EMB')

    # Memory cleanup 
    del train_df, val_df, test_df

## Run 'consolidate_test_metrics'
combined_df = consolidate_test_metrics("logR_cls_t_EMB_model", 30)
print('Consolidated metrics for cls_t_EMB:')
print(combined_df)
print('')

## Run 'consolidate_confusion_matrices'  
consolidate_confusion_matrices(
    model_name_prefix='logR_cls_t_EMB_model',
    n_models=30,
    save_dir=save_dir,
    output_excel_name='confusion_matrices_logR_cls_t_EMB_all.xlsx'
)

print(f"Finished job for cls_t_EMB data at: {datetime.now()}")


# ## aa_EMB

# In[ ]:


# Parameters
N_SPLITS = 30  # Number of sets to run
base_path_linux = '/home/afarodrigues/datasets/Filipa'
base_path_win = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation'

# Output directories
save_dir = '/home/afarodrigues/embCOMP/outputs_csv' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'
models_dir = '/home/afarodrigues/embCOMP/outputs_others' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'

print(f"Started job for aa_EMB data at: {datetime.now()}")

# One-at-a-time loop
for i in range(1, N_SPLITS + 1):
    print(f"\n=== Starting job for split {i} at {datetime.now()} ===")

    # Load only split i
    train_df = load_split_df('train', i, FINAL_RUN, base_path_linux, base_path_win)
    val_df   = load_split_df('val', i, FINAL_RUN, base_path_linux, base_path_win)
    test_df  = load_split_df('test', i, FINAL_RUN, base_path_linux, base_path_win)

    # Run training/evaluation
    model_name = f'logR_aa_EMB_model_{i}'
    train_and_evaluate_logR_definedsplit(train_df, val_df, test_df,
                                       model_name=model_name,
                                       representation='aa_EMB')

    # Memory cleanup 
    del train_df, val_df, test_df

## Run 'consolidate_test_metrics'
combined_df = consolidate_test_metrics("logR_aa_EMB_model", 30)
print('Consolidated metrics for aa_EMB:')
print(combined_df)
print('')

## Run 'consolidate_confusion_matrices'  
consolidate_confusion_matrices(
    model_name_prefix='logR_aa_EMB_model',
    n_models=30,
    save_dir=save_dir,
    output_excel_name='confusion_matrices_logR_aa_EMB_all.xlsx'
)

print(f"Finished job for aa_EMB data at: {datetime.now()}")


# ## hadamard_EMB

# In[ ]:


# Parameters
N_SPLITS = 30  # Number of sets to run
base_path_linux = '/home/afarodrigues/datasets/Filipa'
base_path_win = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation'

# Output directories
save_dir = '/home/afarodrigues/embCOMP/outputs_csv' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'
models_dir = '/home/afarodrigues/embCOMP/outputs_others' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'

print(f"Started job for hadamard_EMB data at: {datetime.now()}")

# One-at-a-time loop
for i in range(1, N_SPLITS + 1):
    print(f"\n=== Starting job for split {i} at {datetime.now()} ===")

    # Load only split i
    train_df = load_split_df('train', i, FINAL_RUN, base_path_linux, base_path_win)
    val_df   = load_split_df('val', i, FINAL_RUN, base_path_linux, base_path_win)
    test_df  = load_split_df('test', i, FINAL_RUN, base_path_linux, base_path_win)

    # Run training/evaluation
    model_name = f'logR_hadamard_EMB_model_{i}'
    train_and_evaluate_logR_definedsplit(train_df, val_df, test_df,
                                       model_name=model_name,
                                       representation='hadamard_EMB')

    # Memory cleanup 
    del train_df, val_df, test_df

## Run 'consolidate_test_metrics'
combined_df = consolidate_test_metrics("logR_hadamard_EMB_model", 30)
print('Consolidated metrics for hadamard_EMB:')
print(combined_df)
print('')

## Run 'consolidate_confusion_matrices'  
consolidate_confusion_matrices(
    model_name_prefix='logR_hadamard_EMB_model',
    n_models=30,
    save_dir=save_dir,
    output_excel_name='confusion_matrices_logR_hadamard_EMB_all.xlsx'
)

print(f"Finished job for hadamard_EMB data at: {datetime.now()}")


# ## L1_EMB

# In[ ]:


# Parameters
N_SPLITS = 30  # Number of sets to run
base_path_linux = '/home/afarodrigues/datasets/Filipa'
base_path_win = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation'

# Output directories
save_dir = '/home/afarodrigues/embCOMP/outputs_csv' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'
models_dir = '/home/afarodrigues/embCOMP/outputs_others' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'

print(f"Started job for L1_EMB data at: {datetime.now()}")

# One-at-a-time loop
for i in range(1, N_SPLITS + 1):
    print(f"\n=== Starting job for split {i} at {datetime.now()} ===")

    # Load only split i
    train_df = load_split_df('train', i, FINAL_RUN, base_path_linux, base_path_win)
    val_df   = load_split_df('val', i, FINAL_RUN, base_path_linux, base_path_win)
    test_df  = load_split_df('test', i, FINAL_RUN, base_path_linux, base_path_win)

    # Run training/evaluation
    model_name = f'logR_L1_EMB_model_{i}'
    train_and_evaluate_logR_definedsplit(train_df, val_df, test_df,
                                       model_name=model_name,
                                       representation='L1_EMB')

    # Memory cleanup 
    del train_df, val_df, test_df

## Run 'consolidate_test_metrics'
combined_df = consolidate_test_metrics("logR_L1_EMB_model", 30)
print('Consolidated metrics for L1_EMB:')
print(combined_df)
print('')

## Run 'consolidate_confusion_matrices'  
consolidate_confusion_matrices(
    model_name_prefix='logR_L1_EMB_model',
    n_models=30,
    save_dir=save_dir,
    output_excel_name='confusion_matrices_logR_L1_EMB_all.xlsx'
)

print(f"Finished job for L1_EMB data at: {datetime.now()}")


# ## L2_EMB

# In[ ]:


# Parameters
N_SPLITS = 30  # Number of sets to run
base_path_linux = '/home/afarodrigues/datasets/Filipa'
base_path_win = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation'

# Output directories
save_dir = '/home/afarodrigues/embCOMP/outputs_csv' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'
models_dir = '/home/afarodrigues/embCOMP/outputs_others' if FINAL_RUN else r'C:\Users\rodri\OneDrive\Desktop\embCOMP\05_LR_classifiers'

print(f"Started job for L2_EMB data at: {datetime.now()}")

# One-at-a-time loop
for i in range(1, N_SPLITS + 1):
    print(f"\n=== Starting job for split {i} at {datetime.now()} ===")

    # Load only split i
    train_df = load_split_df('train', i, FINAL_RUN, base_path_linux, base_path_win)
    val_df   = load_split_df('val', i, FINAL_RUN, base_path_linux, base_path_win)
    test_df  = load_split_df('test', i, FINAL_RUN, base_path_linux, base_path_win)

    # Run training/evaluation
    model_name = f'logR_L2_EMB_model_{i}'
    train_and_evaluate_logR_definedsplit(train_df, val_df, test_df,
                                       model_name=model_name,
                                       representation='L2_EMB')

    # Memory cleanup 
    del train_df, val_df, test_df

## Run 'consolidate_test_metrics'
combined_df = consolidate_test_metrics("logR_L2_EMB_model", 30)
print('Consolidated metrics for L2_EMB:')
print(combined_df)
print('')

## Run 'consolidate_confusion_matrices'  
consolidate_confusion_matrices(
    model_name_prefix='logR_L2_EMB_model',
    n_models=30,
    save_dir=save_dir,
    output_excel_name='confusion_matrices_logR_L2_EMB_all.xlsx'
)

print(f"Finished job for L2_EMB data at: {datetime.now()}")

