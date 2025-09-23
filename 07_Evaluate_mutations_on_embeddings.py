#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

#For OHE
from keras.preprocessing.sequence import pad_sequences

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

#For ProtBERT Encodings
import torch
from transformers import BertModel, BertTokenizer

#For tSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#For parquet 
import pyarrow as pa
import pyarrow.parquet as pq


# In[2]:


# AAV2 VP1 reference reference
aav2capsid_refSeq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

# Amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# ### Auxiliary functions to generate ProtBERT embeddings

# In[3]:


## Function to preprocess amino acid sequences 
def preprocess_sequence(sequence):
    # Capitalize all letters
    sequence = sequence.upper()
    
    # Replace U, Z, O, or B with X
    sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
    
    # Add space between each letter
    sequence = " ".join(sequence)
    
    return sequence

## Function 'get_protBERT_embs', to get all protBERT embeddings
# pick device automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_protBERT_embs(sequence):
    global tokenizer, model
    
    # Load tokenizer if not yet loaded
    if "tokenizer" not in globals():
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    
    # Load model if not yet loaded
    if "model" not in globals():
        model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").to(DEVICE)
        print(f"BertModel loaded on {DEVICE}")
    
    # Tokenize
    encoded_input = tokenizer(sequence, return_tensors="pt").to(DEVICE)

    # Run inference
    with torch.no_grad():
        output = model(**encoded_input)

    return (
        output.last_hidden_state[:, 0, :],               # CLS_raw_EMB
        output.pooler_output,                            # CLS_t_EMB
        output.last_hidden_state[:, 1:-1, :].mean(dim=1) # aa_EMB
    )

## Function 'process_dataframe'
def process_dataframe(df):
    import time
    start_time = time.time()

    # Preprocess sequences
    df['Processed_sequence'] = df['Sequence'].apply(preprocess_sequence)

    # Generate ProtBERT embeddings
    df[['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB']] = df['Processed_sequence'].apply(
        lambda seq: pd.Series([x.cpu().numpy() for x in get_protBERT_embs(seq)])
    )

    # Convert numpy arrays to Python lists
    df['cls_raw_EMB'] = df['cls_raw_EMB'].apply(lambda x: x.tolist())
    df['cls_t_EMB'] = df['cls_t_EMB'].apply(lambda x: x.tolist())
    df['aa_EMB'] = df['aa_EMB'].apply(lambda x: x.tolist())

    # Step 4: Drop intermediate
    df = df.drop(columns=['Processed_sequence'])

    print(f"Execution time: {(time.time() - start_time) / 60:.2f} minutes")
    return df

## Function 'strip_outer_array' to remove the extra (empty) dimension 
def strip_outer_array(df, columns):
    """
    Removes the outer array (first element) from specified columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to process.
    
    Returns:
        pd.DataFrame: The modified DataFrame with updated columns.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: x[0])
    return df


# ### Auxiliary functions to generate OHE

# In[4]:


## Maximum length calculation (for padding)
file_path = r"C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_2.csv"

aav_data = pd.read_csv(file_path)

max_seqLength_AAV_data = aav_data['Sequence'].apply(len).max()
print('maximum length:', max_seqLength_AAV_data)
print('')

# Function 'one_hot_encode_sequences'
def one_hot_encode_sequences(result_df, max_length):
    # Extract sequences from DataFrame
    sequences = result_df['Sequence'].tolist()

    # Convert sequences to list of lists (each character as a list element)
    sequences = [list(seq) for seq in sequences]

    # Pad sequences with 'X' to ensure equal length
    padded_sequences = pad_sequences(sequences, padding='post', truncating='post', value='X', maxlen=max_length, dtype=object)

    # Define the order of amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    all_characters = amino_acids + 'X'  # Include padding character 'X'

    # Create a dictionary to map characters to their index in the one-hot encoding
    char_to_index = {char: i for i, char in enumerate(all_characters)}

    # Initialize a list to store one-hot encoded sequences
    one_hot_sequences = []

    # Iterate over each padded sequence
    for seq in padded_sequences:
        # Initialize a list to store the one-hot encoded binary vector
        one_hot_vector = []

        # Iterate over each character in the sequence
        for char in seq:
            # Initialize a list of zeros with length equal to the number of characters
            one_hot_char = [0] * len(all_characters)
            
            # If the character is valid, set the corresponding index to 1
            if char in char_to_index:
                one_hot_char[char_to_index[char]] = 1
            
            # Append the one-hot encoded character to the binary vector
            one_hot_vector.extend(one_hot_char)

        # Append the one-hot encoded vector as a list to the list of sequences
        one_hot_sequences.append(one_hot_vector)  # Store as list, not formatted string

    # Add the one-hot encoded sequences as a column in the DataFrame
    result_df['OHE'] = one_hot_sequences

    return result_df

## Function 'reduce_OHE_with_svd'
def reduce_with_svd(df, col="OHE", k=1024, random_state=42):
    """
    Apply Truncated SVD + scaling to reduce a vector representation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a column with list/array embeddings.
    col : str, default="OHE"
        Name of the column with representations to reduce.
    k : int, default=1024
        Number of dimensions after reduction.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        Copy of the input dataframe with an additional column
        named f"{col}_SVD" containing the reduced vectors.
    """
    # Convert list of arrays into 2D numpy array
    X = np.array(df[col].to_list())
    
    # Truncated SVD
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    # Save reduced vectors into dataframe
    df = df.copy()
    new_col = f"{col}_SVD"
    df[new_col] = list(X_scaled)
    
    print(f"[{col}] reduced from {X.shape[1]} → {k} dimensions "
          f"(variance explained: {svd.explained_variance_ratio_.sum():.2%})")
    
    return df


# ### Auxiliary functions to genearte sequences with spread mutations

# In[5]:


# Function mutate_sequence_with_log
def mutate_sequence_with_log(seq, x):
    """Generate one mutated sequence with x random mutations, and log events."""
    seq = list(seq)
    log = []
    for _ in range(x):
        mutation_type = random.choice(["substitution", "insertion", "deletion"])
        pos = random.randrange(len(seq))  # position in current sequence
        
        if mutation_type == "substitution":
            old = seq[pos]
            new = random.choice(AMINO_ACIDS)
            seq[pos] = new
            log.append(f"sub@{pos}:{old}->{new}")
        
        elif mutation_type == "insertion":
            new = random.choice(AMINO_ACIDS)
            seq.insert(pos, new)
            log.append(f"ins@{pos}:{new}")
        
        elif mutation_type == "deletion" and len(seq) > 1:
            old = seq.pop(pos)
            log.append(f"del@{pos}:{old}")
    
    return "".join(seq), ";".join(log)

# Function generate_variants_with_log
def generate_variants_with_log(n, x, strategy_name, ref_seq=aav2capsid_refSeq, seed=42):
    """Generate n variants with x mutations each, return DataFrame with logs."""

    records = []
    for i in range(1, n+1):
        mutated, log = mutate_sequence_with_log(ref_seq, x)
        identifier = f"{strategy_name}_{i:02d}"
        records.append((identifier, mutated, log))
    
    df = pd.DataFrame(records, columns=["id", "Sequence", "mutations"])
    return df


# In[6]:


## Generate variants with spread mutations

# 1_mutation variants
df_1 = generate_variants_with_log(100, 1, "1_mutations")

# 5_mutation variants
df_5 = generate_variants_with_log(100, 5, "5_mutations")

# 10_mutation variants
df_10 = generate_variants_with_log(100, 10, "10_mutations")

# 50_mutation variants
df_50 = generate_variants_with_log(100, 50, "50_mutations")

# 100_mutation variants
df_100 = generate_variants_with_log(100, 100, "100_mutations")

# 500_mutation variants
df_500 = generate_variants_with_log(100, 500, "500_mutations")

# Combine all variant DataFrames into one
df_mutations = pd.concat([df_1, df_5, df_10, df_50, df_100, df_500], ignore_index=True)

# Reference row (with no mutations)
ref_row = pd.DataFrame([{
    "id": "reference_00",
    "Sequence": aav2capsid_refSeq,
    "mutations": "none"
}])

# Prepend to df_mutations
df_mutations = pd.concat([ref_row, df_mutations], ignore_index=True)

#Inspect
print(df_mutations)


# ### Generate encodings and embeddings - spread mutations

# In[8]:


## generate OHE
# Record the start time
start_time = time.time()

# Application of 'one_hot_encode_sequences' function to aav_df
df_mutations = one_hot_encode_sequences(df_mutations, max_seqLength_AAV_data)

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = (end_time - start_time)/60

print("Execution time for one hot encodings test set generation:", execution_time, "minutes")
print('Finished the generation of OHEs for test set')
print('')


# In[9]:


## Generate OHE-SVD
df_mutations = reduce_with_svd(df_mutations, col="OHE", k=1024)


# In[10]:


# Generate ProtBERT embeddings for all variants
df_mutations = process_dataframe(df_mutations)

# Remove the outer layer
df_mutations = strip_outer_array(df_mutations, ['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB'])

#Inspect
df_mutations


# In[11]:


## Generate cls_raw_EMB_SVD
df_mutations = reduce_with_svd(df_mutations, col="cls_raw_EMB", k=1024)


# In[12]:


## Generate cls_t_EMB_SVD
df_mutations = reduce_with_svd(df_mutations, col="cls_t_EMB", k=1024)


# In[13]:


## Generate aa_EMB_SVD
df_mutations = reduce_with_svd(df_mutations, col="aa_EMB", k=1024)


# In[14]:


#Function run_tsne_and_plot
def run_tsne_and_plot(
    df, 
    embedding_col="", 
    perplexity=50, 
    random_state=42, 
    save=True, 
    suffix=""
):
    """
    Runs t-SNE on the given embedding column of df and plots with group-based coloring.
    """
    # Extract embeddings
    X = np.array(df[embedding_col].to_list())
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        random_state=random_state, 
        init="pca"
    )
    X_embedded = tsne.fit_transform(X)
    
    # Assign group by removing last "_xx" part from id
    df = df.copy()
    df["group"] = df["id"].str.rsplit("_", n=1).str[0]

    # Plot
    plt.figure(figsize=(8.5, 8))
    for group in df["group"].unique():
        idx = df["group"] == group
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=group, alpha=0.7)

    # Get axis limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Add arrow pointing to reference if it exists
    ref_idx = df["id"].str.contains("reference", case=False, na=False)
    if ref_idx.any():
        ref_x, ref_y = X_embedded[ref_idx, 0][0], X_embedded[ref_idx, 1][0]

        # Offset = 5% of plot range
        offset_x = 0.05 * x_range
        offset_y = 0.05 * y_range

        # Flip offsets if too close to edge
        if ref_x + offset_x > x_max:
            offset_x = -offset_x
        if ref_x + offset_x < x_min:
            offset_x = abs(offset_x)
        if ref_y + offset_y > y_max:
            offset_y = -offset_y
        if ref_y + offset_y < y_min:
            offset_y = abs(offset_y)

        plt.annotate(
            "",
            xy=(ref_x, ref_y),
            xytext=(ref_x + offset_x, ref_y + offset_y),
            arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5),
            fontsize=12,
            fontweight="bold"
        )
    
    plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"t-SNE of {embedding_col}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Save figure
    if save:
        fname = f"t-SNE group comparison {embedding_col} {suffix}.png".strip()
        plt.savefig(fname, dpi=600, bbox_inches="tight")
        print(f"Figure saved as {fname}")
    
    plt.show()
    
    return X_embedded


# In[15]:


tSNE = run_tsne_and_plot(df_mutations, embedding_col="cls_raw_EMB", suffix="spread mutations")


# In[16]:


tSNE = run_tsne_and_plot(df_mutations, embedding_col="cls_raw_EMB_SVD", suffix="spread mutations")


# In[17]:


tSNE = run_tsne_and_plot(df_mutations, embedding_col="cls_t_EMB", suffix="spread mutations")


# In[18]:


tSNE = run_tsne_and_plot(df_mutations, embedding_col="cls_t_EMB_SVD", suffix="spread mutations")


# In[19]:


tSNE = run_tsne_and_plot(df_mutations, embedding_col="aa_EMB", suffix="spread mutations")


# In[20]:


tSNE = run_tsne_and_plot(df_mutations, embedding_col="aa_EMB_SVD", suffix="spread mutations")


# In[21]:


tSNE = run_tsne_and_plot(df_mutations, embedding_col="OHE", suffix="spread mutations")


# In[22]:


tSNE = run_tsne_and_plot(df_mutations, embedding_col="OHE_SVD", suffix="spread mutations")


# ## Auxiliary functions to generate localized mutations

# In[23]:


## Function 'mutate_region'
def mutate_region_variants(reference_seq, start, end, n_variants=100, seed=42):
    """
    Generate n mutated variants targeting a region [start, end].
    Each position in the region undergoes substitution, insertion, or deletion.
    
    Args:
        reference_seq (str): Original sequence
        start (int): Start position (1-based, inclusive)
        end (int): End position (1-based, inclusive)
        n_variants (int): Number of variants to generate
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame with [ID, Sequence]
    """
    
    variants = []
    
    for i in range(1, n_variants + 1):
        seq = list(reference_seq)
        mutated_seq = []
        
        for j, aa in enumerate(seq, start=1):
            if start <= j <= end:
                mutation_type = random.choice(["sub", "ins", "del"])
                
                if mutation_type == "sub":
                    # substitution
                    new_aa = random.choice([x for x in AMINO_ACIDS if x != aa])
                    mutated_seq.append(new_aa)
                
                elif mutation_type == "ins":
                    # insertion before this position
                    ins_aa = random.choice(AMINO_ACIDS)
                    mutated_seq.append(ins_aa)
                    mutated_seq.append(aa)
                
                elif mutation_type == "del":
                    # deletion
                    continue
            else:
                mutated_seq.append(aa)
        
        mutated_seq = "".join(mutated_seq)
        variant_id = f"fragment_{start}to{end}_{i:02d}"
        variants.append((variant_id, mutated_seq))
    
    return pd.DataFrame(variants, columns=["id", "Sequence"])


# In[24]:


## Generate variants

# region_560_588 variants (28 aa region)
region_561_588 = mutate_region_variants(aav2capsid_refSeq, 561, 588, n_variants=100, seed=42)

# region_540_608 variants (68 aa region)
region_541_608 = mutate_region_variants(aav2capsid_refSeq, 541, 608, n_variants=100, seed=42)

# region_520_628 variants (128 aa region)
region_521_628 = mutate_region_variants(aav2capsid_refSeq, 521, 628, n_variants=100, seed=42)

# region_490_648 variants (168 aa region)
region_491_648 = mutate_region_variants(aav2capsid_refSeq, 491, 648, n_variants=100, seed=42)

# region_470_668 variants (208 aa region)
region_471_668 = mutate_region_variants(aav2capsid_refSeq, 471, 668, n_variants=100, seed=42)

# region_450_688 variants (248 aa region)
region_451_688 = mutate_region_variants(aav2capsid_refSeq, 451, 688, n_variants=100, seed=42)

# Combine all variant DataFrames into one
df_regions = pd.concat([region_561_588, region_541_608, region_521_628, region_491_648, region_471_668, region_451_688], ignore_index=True)

# Reference row (with no mutations)
ref_row = pd.DataFrame([{
    "id": "reference_00",
    "Sequence": aav2capsid_refSeq,
}])

# Prepend to df_regions
df_regions = pd.concat([ref_row, df_regions], ignore_index=True)
df_regions


# In[25]:


# Add a new column with sequence lengths
df_regions["seq_length"] = df_regions["Sequence"].str.len()

## Function 'longest_matching_prefix_len' to check if the targeting region is correct
def longest_matching_prefix_len(seq, ref=aav2capsid_refSeq):
    if pd.isna(seq):
        return pd.NA
    seq = str(seq)
    m = min(len(seq), len(ref))
    i = 0
    # count how many leading positions match
    while i < m and seq[i] == ref[i]:
        i += 1
    return i  # 0..m  (1-based position of last matching residue)

# Add the column
df_regions["unchanged_until"] = df_regions["Sequence"].apply(longest_matching_prefix_len).astype("Int64")

# Inspect
df_regions


# ### Generate encodings and embeddings - localized mutations

# In[26]:


## generate OHE for localized mutations
# Record the start time
start_time = time.time()

# Application of 'one_hot_encode_sequences' function to aav_df
df_regions = one_hot_encode_sequences(df_regions, max_seqLength_AAV_data)

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = (end_time - start_time)/60

print("Execution time for one hot encodings generation:", execution_time, "minutes")
print('Finished the generation of OHEs')
print('')


# In[27]:


## Generate OHE-SVD for localized mutations
df_regions = reduce_with_svd(df_regions, col="OHE", k=1024)


# In[28]:


# Apply ProtBERT embeddings to all variants
df_regions = process_dataframe(df_regions)

# Remove the outer layer
df_regions = strip_outer_array(df_regions, ['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB'])

#Inspect
print(df_regions)


# In[29]:


## Generate cls_raw_EMB-SVD for localized mutations
df_regions = reduce_with_svd(df_regions, col="cls_raw_EMB", k=1024)


# In[30]:


## Generate cls_t_EMB-SVD for localized mutations
df_regions = reduce_with_svd(df_regions, col="cls_t_EMB", k=1024)


# In[31]:


## Generate aa_EMB-SVD for localized mutations
df_regions = reduce_with_svd(df_regions, col="aa_EMB", k=1024)


# In[32]:


tSNE = run_tsne_and_plot(df_regions, embedding_col="cls_raw_EMB",  suffix="localized mutations")


# In[33]:


tSNE = run_tsne_and_plot(df_regions, embedding_col="cls_raw_EMB_SVD",  suffix="localized mutations")


# In[34]:


tSNE = run_tsne_and_plot(df_regions, embedding_col="cls_t_EMB",  suffix="localized mutations")


# In[35]:


tSNE = run_tsne_and_plot(df_regions, embedding_col="cls_t_EMB_SVD",  suffix="localized mutations")


# In[36]:


tSNE = run_tsne_and_plot(df_regions, embedding_col="aa_EMB",  suffix="localized mutations")


# In[37]:


tSNE = run_tsne_and_plot(df_regions, embedding_col="aa_EMB_SVD",  suffix="localized mutations")


# In[38]:


tSNE = run_tsne_and_plot(df_regions, embedding_col="OHE",  suffix="localized mutations")


# In[39]:


tSNE = run_tsne_and_plot(df_regions, embedding_col="OHE_SVD",  suffix="localized mutations")


# In[ ]:




