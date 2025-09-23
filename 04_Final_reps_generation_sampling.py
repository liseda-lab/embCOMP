
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, vstack

import os

from sklearn.model_selection import train_test_split

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


# In[2]:


# Gate for CLUSTER run
CLUSTER_RUN = True  # Set to False to run locally

if CLUSTER_RUN:
    print('CLUSTER RUN: Running in Cluster')
else:
    print('LOCAL RUN: Run locally')


# ### Load OHE data

# In[ ]:


## Train data

# Load non-OHE columns from the Parquet file
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_train_metadata.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_train_metadata.parquet'

metadata = pd.read_parquet(file_path)

# Load the OHE vectors from the NumPy file
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_train.npy'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_train.npy'

ohe_matrix = np.load(file_path, allow_pickle=True)

# Convert each row to sparse format immediately
sparse_ohe_matrix = [csr_matrix(row) for row in ohe_matrix]

del ohe_matrix

# Add the sparse OHE matrix back as a new column in the DataFrame
metadata['OHE'] = sparse_ohe_matrix

# Rename
ohe_train = metadata

# Inspect
print(ohe_train.head(5))
print('\nFinished loading of OHE train set\n')


# In[ ]:


## Test val

# Load non-OHE columns from the Parquet file
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_val_metadata.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_val_metadata.parquet'

metadata = pd.read_parquet(file_path)

# Load the OHE vectors from the NumPy file
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_val.npy'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_val.npy'

ohe_matrix = np.load(file_path, allow_pickle=True)

# Convert each row to sparse format immediately
sparse_ohe_matrix = [csr_matrix(row) for row in ohe_matrix]

del ohe_matrix

# Add the sparse OHE matrix back as a new column in the DataFrame
metadata['OHE'] = sparse_ohe_matrix

# Rename
ohe_val = metadata

# Inspect
print(ohe_val.head(5))
print('\nFinished loading of OHE val set\n')


# In[ ]:


## Test data

# Load non-OHE columns from the Parquet file
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_test_metadata.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_test_metadata.parquet'

metadata = pd.read_parquet(file_path)

# Load the OHE vectors from the NumPy file
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_test.npy'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_test.npy'

ohe_matrix = np.load(file_path, allow_pickle=True)

# Convert each row to sparse format immediately
sparse_ohe_matrix = [csr_matrix(row) for row in ohe_matrix]

del ohe_matrix

# Add the sparse OHE matrix back as a new column in the DataFrame
metadata['OHE'] = sparse_ohe_matrix

# Rename
ohe_test = metadata

# Inspect
print(ohe_test.head(5))
print('\nFinished loading of OHE test set\n')


# In[ ]:


# Combine all into one DataFrame
full_df = pd.concat([ohe_train, ohe_val, ohe_test], axis=0).reset_index(drop=True)

# Extract labels
y = full_df['Viability']

# First: train vs temp (train = 70%)
train_df, temp_df = train_test_split(
    full_df,
    train_size=0.7,
    stratify=y,
    random_state=30
)

# Second: val vs test (val = 10%, test = 20% → ratio = 1:2)
y_temp = temp_df['Viability']
val_df, test_df = train_test_split(
    temp_df,
    train_size=1/3,  # because 10 / (10 + 20)
    stratify=y_temp,
    random_state=42
)

# Assign to ohe_* variables
ohe_train = train_df.reset_index(drop=True)
ohe_val = val_df.reset_index(drop=True)
ohe_test = test_df.reset_index(drop=True)

#Free memory
del full_df

#Inspect
print('OHE test set 30:')
print(ohe_test.head(5)) 
print('')


# In[ ]:


## Apply truncated SVD
## Note: if SVD is independently applied  on each split, each will have its own internal structure, which invalidates comparisons
## Therefore, fit SVD only on the training set and then transform the validation and test sets using that trained decomposition

# Stack OHE arrays
X_train_raw = vstack(ohe_train['OHE'].values).tocsr()
X_val_raw = vstack(ohe_val['OHE'].values).tocsr()
X_test_raw = vstack(ohe_test['OHE'].values).tocsr()

# Clean up
del ohe_train, ohe_val, ohe_test

# Apply SVD
svd = TruncatedSVD(n_components=1024, random_state=42)
X_train_svd = svd.fit_transform(X_train_raw)
X_val_svd   = svd.transform(X_val_raw)
X_test_svd  = svd.transform(X_test_raw)

# Clean up
del X_train_raw, X_val_raw, X_test_raw

# Apply scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_svd)
X_val   = scaler.transform(X_val_svd)
X_test  = scaler.transform(X_test_svd)

del X_train_svd, X_val_svd, X_test_svd


# ### Auxiliary functions

# In[ ]:


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


# In[ ]:


# Function 'compute_vector_combinations' to generate new columns with hadamard, L1 and L2
def compute_vector_combinations(df, vec1_col, vec2_col, prefix=''):
    """
    Computes Hadamard product, L1 distance, and L2 distance between two vector columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        vec1_col (str): Name of the first vector column.
        vec2_col (str): Name of the second vector column.
        prefix (str): Optional prefix for new column names.

    Returns:
        pd.DataFrame: The modified DataFrame with new columns added.
    """
    df[f'{prefix}hadamard_EMB'] = df.apply(lambda row: row[vec1_col] * row[vec2_col], axis=1)
    df[f'{prefix}L1_EMB'] = df.apply(lambda row: np.abs(row[vec1_col] - row[vec2_col]), axis=1)
    df[f'{prefix}L2_EMB'] = df.apply(lambda row: (row[vec1_col] - row[vec2_col]) ** 2, axis=1)
    return df


# In[ ]:


## Function 'check_consistent_dimensions' to check embs dimensions
def check_consistent_dimensions(df, columns):
    """
    Prints the unique shapes or lengths of arrays/sequences in specified DataFrame columns
    to confirm dimensional consistency.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to inspect.
    """
    for col in columns:
        lengths = df[col].apply(lambda x: getattr(x, 'shape', (len(x),)))
        unique_lengths = lengths.drop_duplicates()
        print(f"{col}: unique shapes/lengths: {unique_lengths.tolist()}")


# ### Generate final representations - EMB representations

# In[ ]:


#load embs data - train set

#file paths
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/datasets/Filipa/protB_EMB_train.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\protB_EMB_train.parquet'

train = pd.read_parquet(file_path)

# Apply 'strip_outer_arry' to remove the extra dimension
train = strip_outer_array(train, ['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB'])

# Apply 'compute_vector_combinations' to generate combined embeddings
train = compute_vector_combinations(train, 'cls_raw_EMB', 'aa_EMB')

# Apply 'check_consistent_dimensions' function
check_consistent_dimensions(train, ['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB','hadamard_EMB', 'L1_EMB', 'L2_EMB' ])


# In[ ]:


#load embs data - validation set

#file paths
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/datasets/Filipa/protB_EMB_val.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\protB_EMB_val.parquet'

val = pd.read_parquet(file_path)

# Apply 'strip_outer_arry' to remove the extra dimension
val = strip_outer_array(val, ['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB'])

# Apply 'compute_vector_combinations' to generate combined embeddings
val = compute_vector_combinations(val, 'cls_raw_EMB', 'aa_EMB')

# Apply 'check_consistent_dimensions' function
check_consistent_dimensions(val, ['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB','hadamard_EMB', 'L1_EMB', 'L2_EMB' ])


# In[ ]:


#load embs data - test set

#file paths
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/datasets/Filipa/protB_EMB_test.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\protB_EMB_test.parquet'

test = pd.read_parquet(file_path)

# Apply 'strip_outer_arry' to remove the extra dimension
test = strip_outer_array(test, ['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB'])

# Apply 'compute_vector_combinations' to generate combined embeddings
test = compute_vector_combinations(test, 'cls_raw_EMB', 'aa_EMB')

# Apply 'check_consistent_dimensions' function
check_consistent_dimensions(test, ['cls_raw_EMB', 'cls_t_EMB', 'aa_EMB','hadamard_EMB', 'L1_EMB', 'L2_EMB' ])


# In[ ]:


# Combine all into one DataFrame
full_df = pd.concat([train, val, test], axis=0).reset_index(drop=True)

# Extract labels
y = full_df['Viability']

# First: train vs temp (train = 70%)
train_df, temp_df = train_test_split(
    full_df,
    train_size=0.7,
    stratify=y,
    random_state=30
)

# Second: val vs test (val = 10%, test = 20% → ratio = 1:2)
y_temp = temp_df['Viability']
val_df, test_df = train_test_split(
    temp_df,
    train_size=1/3,  # because 10 / (10 + 20)
    stratify=y_temp,
    random_state=42
)


# ### Add OHE and save

# In[ ]:


## Train

# Assign to ohe_* variables
train = train_df.reset_index(drop=True)

# Insert the column [OHE-SVD] from X_train_scaled at position 5 (6th column, 0-indexed)
train.insert(5, 'OHE-SVD', list(X_train))

#Save
#paths
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/datasets/Filipa/final_REPs_train_30.parquet'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\final_REPs_train_30.parquet'

train.to_parquet(save_path)

#Inspect
print('Final REPs for train set 30:')
print(train.head(5)) 
print('')


# In[ ]:


## Val

# Assign to ohe_* variables
val = val_df.reset_index(drop=True)

# Insert the column [OHE-SVD] from X_train_scaled at position 5 (6th column, 0-indexed)
val.insert(5, 'OHE-SVD', list(X_val))

#Save
#paths
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/datasets/Filipa/final_REPs_val30.parquet'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\final_REPs_val_30.parquet'

val.to_parquet(save_path)

#Inspect
print('Final REPs for validation set 30:')
print(val.head(5)) 
print('')


# ### Generate final representations - Test set

# In[ ]:





# In[ ]:


## Test 

# Assign to ohe_* variables
test = test_df.reset_index(drop=True)

# Insert the column [OHE-SVD] from X_train_scaled at position 5 (6th column, 0-indexed)
test.insert(5, 'OHE-SVD', list(X_test))

#Save
#paths
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/datasets/Filipa/final_REPs_test_30.parquet'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\final_REPs_test_30.parquet'

test.to_parquet(save_path)

#Inspect
print('Final REPs for test set 30:')
print(test.head(5)) 
print('')


# In[ ]:



