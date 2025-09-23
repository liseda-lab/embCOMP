#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import numpy as np

from Bio import SeqIO
import random

import time
from datetime import datetime

from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle


# In[ ]:


# Gate for final run
FINAL_RUN = True  # Set to False for testing/dev mode

if FINAL_RUN:
    print('FINAL RUN: Using full dataset and production file paths')
else:
    print('TEST MODE: Sampling dataset and using local/test file paths')


# ### Load dataset

# In[2]:


# Train
if FINAL_RUN:
    file_path = r'/home/afarodrigues/datasets/Filipa/data_train.csv'  
else:
    file_path = r"C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_train.csv"

ohe_data_train = pd.read_csv(file_path)

# Validation
if FINAL_RUN:
    file_path = r'/home/afarodrigues/datasets/Filipa/data_val.csv'
else:
    file_path = r"C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_val.csv"

ohe_data_val = pd.read_csv(file_path)

# Test
if FINAL_RUN:
    file_path = r'/home/afarodrigues/datasets/Filipa/data_test.csv'
else:
    file_path = r"C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_test.csv"

ohe_data_test = pd.read_csv(file_path)

print('Finished datasets loading')
print('')


# ### Auxiliary functions for one hot encoding

# In[4]:


## Maximum length calculation (for padding)

# All data
if FINAL_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_csv/data_2.csv'
else:
    file_path = r"C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_2.csv"

aav_data = pd.read_csv(file_path)

max_seqLength_AAV_data = aav_data['Sequence'].apply(len).max()
print('maximum length:', max_seqLength_AAV_data)
print('')


# In[5]:


# Define the function 'one_hot_encode_sequences'
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

    # Drop the original 'Sequence' column if no longer needed
    result_df = result_df.drop(columns=['Sequence'])

    return result_df


# ### Generate one hot encodings

# In[6]:


## Test set

# Record the start time
start_time = time.time()

# Application of 'one_hot_encode_sequences' function to aav_df
trainVal = one_hot_encode_sequences(ohe_data_test, max_seqLength_AAV_data)

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = (end_time - start_time)/60

print("Execution time for one hot encodings test set generation:", execution_time, "minutes")
print('Finished the generation of OHEs for test set')
print('')


# In[10]:


## Train set

# Record the start time
start_time = time.time()

# Application of 'one_hot_encode_sequences' function to aav_df
trainVal = one_hot_encode_sequences(ohe_data_train, max_seqLength_AAV_data)

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = (end_time - start_time)/60

print("Execution time for one hot encodings train set generation:", execution_time, "minutes")
print('Finished the generation of OHEs for train set')
print('')


# In[7]:


## Validation set

# Record the start time
start_time = time.time()

# Application of 'one_hot_encode_sequences' function to aav_df
trainVal = one_hot_encode_sequences(ohe_data_val, max_seqLength_AAV_data)

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = (end_time - start_time)/60

print("Execution time for one hot encodings validation set generation:", execution_time, "minutes")
print('Finished the generation of OHEs for validation set')
print('')


# In[8]:


# Check OHE vectors length - test set
lengths = ohe_data_test['OHE'].apply(len)
print('lengths for test set:')
print(lengths)
print('')


# In[12]:


# Check OHE vectors length - train set
lengths = ohe_data_train['OHE'].apply(len)
print('lengths for train set:')
print(lengths)
print('')


# In[9]:


# Check OHE vectors length - validation set
lengths = ohe_data_val['OHE'].apply(len)
print('lengths for validation set:')
print(lengths)
print('')


# In[13]:


# Saving strategy with separation of OHE column - test set

# Drop the OHE column from the df
ohe_metadata_dropped = ohe_data_test.drop(columns=['OHE'])

# Save the df to Parquet (without OHE column)
#Save paths
if FINAL_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_test_metadata.parquet'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_test_metadata.parquet'

ohe_metadata_dropped.to_parquet(save_path)

# Save the OHE column as a NumPy array
#Save paths
if FINAL_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_test.npy'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_test.npy'
    
np.save(save_path, ohe_data_test['OHE'].values)

print('Finished split save strategy for test set')
print('')


# In[ ]:


# Saving strategy with separation of OHE column - train set

# Drop the OHE column from the df
ohe_metadata_dropped = ohe_data_train.drop(columns=['OHE'])

# Save the df to Parquet (without OHE column)
#Save paths
if FINAL_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_train_metadata.parquet'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_train_metadata.parquet'
    
ohe_metadata_dropped.to_parquet(save_path)

# Save the OHE column as a NumPy array
#Save paths
if FINAL_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_train.npy'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_train.npy'

np.save(save_path, ohe_data_train['OHE'].values)

print('Finished split save strategy for train set')
print('')


# In[15]:


## Saving strategy with separation of OHE column - validation set

# Drop the OHE column from the df
ohe_metadata_dropped = ohe_data_val.drop(columns=['OHE'])

# Save the df to Parquet (without OHE column)
#Save paths
if FINAL_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_val_metadata.parquet'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ohe_val_metadata.parquet'
    
ohe_metadata_dropped.to_parquet(save_path)

# Save the OHE column as a NumPy array
#Save paths
if FINAL_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_val.npy'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_val.npy'
    
np.save(save_path, ohe_data_val['OHE'].values)

print('Finished split save strategy for validation set')
print('')


# In[16]:


## Verify loading strategy for test set

# Load non-OHE columns from the Parquet file
#File paths
if FINAL_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_test_metadata.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_test_metadata.parquet'

metadata = pd.read_parquet(file_path)

# Load the OHE vectors from the NumPy file
#File paths
if FINAL_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_test.npy'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_test.npy'
    
ohe_matrix = np.load(file_path, allow_pickle=True)

# Add the OHE matrix back as a new column in the DataFrame
metadata['OHE'] = list(ohe_matrix)  # Convert array to list of lists for df compatibility

# Rename 
ohe_test = metadata

# Inspect
print(ohe_test.head(5))
print('')
print('Finished verification loading strategy for test set')
print('')


# In[ ]:


## Verify loading strategy for train set

# Load non-OHE columns from the Parquet file
#File paths
if FINAL_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_train_metadata.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_train_metadata.parquet'

metadata = pd.read_parquet(file_path)

# Load the OHE vectors from the NumPy file
#File paths
if FINAL_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_train.npy'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_train.npy'
    
ohe_matrix = np.load(file_path, allow_pickle=True)

# Add the OHE matrix back as a new column in the DataFrame
metadata['OHE'] = list(ohe_matrix)  # Convert array to list of lists for df compatibility

# Rename 
ohe_train = metadata

# Inspect
print(ohe_train.head(5))
print('')
print('Finished verification loading strategy for train set')
print('')


# In[17]:


## Verify loading strategy for validation set

# Load non-OHE columns from the Parquet file
#File paths
if FINAL_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_val_metadata.parquet'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_val_metadata.parquet'

metadata = pd.read_parquet(file_path)

# Load the OHE vectors from the NumPy file
#File paths
if FINAL_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_others/ohe_val.npy'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\ohe_val.npy'
    
ohe_matrix = np.load(file_path, allow_pickle=True)

# Add the OHE matrix back as a new column in the DataFrame
metadata['OHE'] = list(ohe_matrix)  # Convert array to list of lists for df compatibility

# Rename 
ohe_val = metadata

# Inspect
print(ohe_val.head(5))
print('')
print('Finished verification loading strategy for validation set')
print('')


# In[ ]:




