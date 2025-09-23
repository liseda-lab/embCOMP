#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import re
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Gate for local/cluster
CLUSTER_RUN = True 

if CLUSTER_RUN:
    print('CLUSTER RUN')
else:
    print('LOCAL RUN')


# ## Load datasets

# In[2]:


# Read the CSV file into a DataFrame

if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_csv/data_2.csv'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_2.csv'

aav_df = pd.read_csv(file_path)
aav_df


# #### Split based on viability

# In[3]:


## Viable sequences
viable_df = aav_df[aav_df['Viability'] == 1]  # Rows where Viability is 1

#Export to csv
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_csv/Viable_set.csv'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\Viable_set.csv'

viable_df.to_csv(save_path)

#Inspect
print(viable_df.head(5))


# In[4]:


#Non-viable sequences
non_viable_df = aav_df[aav_df['Viability'] == 0]  # Rows where Viability is 0

#Export to csv
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_csv/Nonviable_set.csv'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\Nonviable_set.csv'

non_viable_df.to_csv(save_path)

#Inspect
print(non_viable_df.head(5))


# #### Split based on design strategy

# In[5]:


# Define categories based on subset
ml_designed_categories = [
    'lr_designed_plus_rand_train_walked', 'lr_rand_doubles_plus_single_walked',
    'rnn_standard_walked', 'cnn_rand_doubles_plus_single_walked', 
    'cnn_standard_walked', 'rnn_designed_plus_rand_train_walked', 
    'rnn_rand_doubles_plus_singles_seed', 'cnn_designed_plus_rand_train_seed',
    'rnn_designed_plus_rand_train_seed', 'lr_standard_seed', 
    'cnn_rand_doubles_plus_single_seed', 'cnn_standard_seed', 
    'cnn_designed_plus_rand_train_seed', 'lr_designed_plus_rand_train_seed', 
    'rnn_standard_seed', 'lr_rand_doubles_plus_single_seed', 
    'lr_standard_walked', 'rnn_rand_doubles_plus_singles_walked',
    'cnn_designed_plus_rand_train_walked'
]

nonml_designed_categories = [
    'designed', 'rand', 'random_doubles', 'single', 'singles', 'stop',
    'previous_chip_viable', 'previous_chip_nonviable'
]

# Create the filtered dfs
ml_designed_df = aav_df[aav_df['Subset'].isin(ml_designed_categories)].copy()
nonml_designed_df = aav_df[aav_df['Subset'].isin(nonml_designed_categories)].copy()


#Export to csv
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_csv/Nonml_designed_set.csv'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\Nonml_designed_set.csv'

#Export to csv
nonml_designed_df.to_csv(save_path)


#Export to csv
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_csv/Ml_designed_set.csv'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\Ml_designed_set.csv'

ml_designed_df.to_csv(save_path)


# ### Main Function: detect_changes(s) (By Lucas Ferraz)

# In[6]:


def detect_changes(s):
    """
    Analyzes the result of a pairwise sequence alignment and returns a list of differences 
    between the mutated and original sequences.

    Parameters:
    s (str): A string containing the output of a pairwise alignment in the form of three lines:
             - mutated sequence
             - alignment matches (using '|' for matches, ' ' for mismatches/gaps)
             - original sequence
             followed by two unused lines (usually empty).

    Returns:
    List[List]: A list of changes detected in the following formats:
                - ["Ins", aa, pos]             → Insertion of amino acid 'aa' at position 'pos'
                - ["Del", aa, pos]             → Deletion of amino acid 'aa' from position 'pos'
                - ["Sub", old_aa, new_aa, pos] → Substitution of 'old_aa' with 'new_aa' at position 'pos'

    Notes:
    - The function merges adjacent insertions and deletions into substitutions when possible.
    - Positions are based on the index in the original sequence (ignoring gaps).
    """

    # The algorithm considers 3 strings: mutated, matches, and original (the 3 lines from the alignment)
    mutated, matches, original, _, _ = s.split("\n")
    
    # This algorithm switches "mode" under certain conditions
    mode = None
    
    # The algorithm needs to remember previous insertions/deletions in order to merge them into substitutions
    # It uses two stacks to "remember" what it read previously
    insert_stack = []
    delete_stack = []
    
    # List to store the final results
    results = []
    
    # Counter for the "real" position in the original string (ignoring gaps)
    c = 0
    
    # For each position in the alignment match line
    for i, m in enumerate(matches):

        # If there is a match at this position (i.e., no change)
        if m == "|":
            
            # Record any previously read deletions
            for elem in delete_stack:
                aa, pos = elem
                results.append(["Del", aa, pos])
                
            # Record any previously read insertions
            for elem in insert_stack:
                aa, pos = elem
                results.append(["Ins", aa, pos])
                
            # Clear the stacks as they have now been processed
            delete_stack = []
            insert_stack = []
            
            # Reset mode to Inactive
            mode = None
        
        # Determine the current mode based on what's at position i
        
        # If the original has an amino acid and mutated has a gap → deletion
        if original[i] != "-" and mutated[i] == "-":
            mode = "Del"
        
        # If the original has a gap and mutated has an amino acid → insertion
        if original[i] == "-" and mutated[i] != "-":
            mode = "Ins"

        # Perform actions based on the current mode
        if mode == "Del":
            if len(insert_stack) > 0:
                # Merge with previous insertion → substitution
                aa, index = insert_stack.pop()
                results.append(["Sub", original[i], aa, c])
            else:
                # Record deletion
                delete_stack.append([original[i], c])
        
        elif mode == "Ins":
            if len(delete_stack) > 0:
                # Merge with previous deletion → substitution
                aa, index = delete_stack.pop()
                results.append(["Sub", aa, mutated[i], c])
            else:
                # Record insertion
                insert_stack.append([mutated[i], c])
        
        # Advance the real position counter if not reading a gap in the original
        if original[i] != "-":
            c += 1
            
    # Final cleanup: write any remaining insertions or deletions to results
    for elem in delete_stack:
        aa, pos = elem
        results.append(["Del", aa, pos])
    for elem in insert_stack:
        aa, pos = elem
        results.append(["Ins", aa, pos - 1])
    
    # Reset (good practice)
    delete_stack = []
    insert_stack = []
    mode = None
    
    # Return sorted list of operations by position
    return sorted(results, key=lambda x: x[-1])


# ### Create 'changes matrix' for each subset

# #### Auxiliary functions 

# In[7]:


# AAV2 VP1 reference reference
aav2vp1_refSeq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

## VP1 changes matrix generation
def make_changes_matrix(df, refSeq):
    # Initialize an empty NumPy array with 4 rows (Unchanged, Sub, Ins, Del) and N columns (length of the reference sequence)
    array = np.zeros((4, len(refSeq)), dtype=int)
    
    # Loop through each sequence in the df
    for index, row in df.iterrows():
        # Access the 'Sequence_Label' and 'Aligned_Piece' columns
        tag = row['Sequence_ID']
        sequence = row['Sequence']
        
        # Perform pairwise alignment with the reference sequence
        alignments = pairwise2.align.globalxx(sequence, refSeq, one_alignment_only=True)
        
        # Format and get the changes for the current sequence
        s = format_alignment(*alignments[0])
        changes = detect_changes(s)
        
        # Initialize a matrix for the current sequence
        matrix = np.zeros((4, len(refSeq)), dtype=int)
        
        # Define unchanged positions
        unchanged_positions = set(range(len(refSeq)))
        
        # Populate the matrix based on the changes
        for change_type, *rest in changes:
            if change_type == "Ins":
                matrix[2][rest[1]] = 1
                unchanged_positions.discard(rest[1])
            elif change_type == "Del":
                matrix[3][rest[1]] = 1
                unchanged_positions.discard(rest[1])
            elif change_type == "Sub":
                matrix[1][rest[2]] = 1
                unchanged_positions.discard(rest[2])
        
        # Populate the Unchanged row
        matrix[0, list(unchanged_positions)] = 1
        
        # Add the current matrix to the result_array
        array += matrix
    
    return array


# #### Change matrices for Viable sequences

print('Started making changes matrix for Viable sequences')


## Application to viable_df
aav_viable = make_changes_matrix(viable_df, refSeq=aav2vp1_refSeq)

# Save the df to a CSV file
df_result = pd.DataFrame(aav_viable)

if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_csv/ChangesMatrix_viable_set.csv'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ChangesMatrix_viable_set.csv'
    
df_result.to_csv(save_path, index=False)

# Print the columns including the 561-588 (fragment) region
print('Result matrix (columns 557 to 590):')
print(df_result.iloc[:, 558:590])  
print('')


# #### Change matrices for Non-viable sequences

print('Started making changes matrix for Non-viable sequences')


## Application to non-viable_df
aav_non_viable = make_changes_matrix(non_viable_df, refSeq=aav2vp1_refSeq)

# Save the df to a CSV file
df_result = pd.DataFrame(aav_non_viable)

if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_csv/ChangesMatrix_nonviable_set.csv'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ChangesMatrix_nonviable_set.csv'
    
df_result.to_csv(save_path, index=False)

# Print the columns including the 561-588 (fragment) region
print('Result matrix (columns 557 to 590):')
print(df_result.iloc[:, 558:590])  
print('')


# #### Change matrices for NonML-desinged sequences

print('Started making changes matrix for NonML-designed sequences')


## Application to nonml-designed_df
aav_nonml = make_changes_matrix(nonml_designed_df, refSeq=aav2vp1_refSeq)

# Save the df to a CSV file
df_result = pd.DataFrame(aav_nonml)

if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_csv/ChangesMatrix_nonml_set.csv'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ChangesMatrix_nonml_set.csv'

df_result.to_csv(save_path, index=False)

# Print the columns including the 561-588 (fragment) region
print('Result matrix (columns 557 to 590):')
print(df_result.iloc[:, 558:590])  
print('')


# #### Change matrices for ML-desinged sequences

print('Started making changes matrix for ML-designed sequences')


## Application to nonml-designed_df
aav_ml = make_changes_matrix(ml_designed_df, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(aav_ml)

if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_csv/ChangesMatrix_ml_set.csv'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ChangesMatrix_ml_set.csv'

df_result.to_csv(save_path, index=False)

# Print the columns including the 561-588 (fragment) region
print('Result matrix (columns 557 to 590):')
print(df_result.iloc[:, 558:590])  
print('')

print('Finished making changes matrices')

# ## Mutation landscape analysis

# ### Auxiliary functions for plotting mutation landscape

# In[12]:


### Function to plot mutation type across positions
def plot_mutation_type_distribution(df, suffix):
    # Calculate percentages for each mutation type at each position
    total_mutations = df.iloc[-1]  # Total mutations per column
    mutation_types = df.iloc[:-1] / total_mutations  # Divide each mutation type count by total mutations
    mutation_types = mutation_types.transpose() * 100  # Convert to percentages and transpose for plotting

    # Create a figure with custom size
    fig, ax = plt.subplots(figsize=(6, 5))  # Adjust the figsize parameter if needed

    # Define custom colors for stacks
    colors = ['blue', 'green', 'red', 'white']

    # Reverse the order of mutation types for stacking
    mutation_types = mutation_types.iloc[:, ::-1]

    # Plot theme river with custom colors and reversed stacking order
    stacks = ax.stackplot(range(len(mutation_types)), mutation_types.values.T, labels=mutation_types.columns, colors=colors)
    ax.set_xlabel('Amino acid position')
    ax.set_ylabel('Percentage of change')
    ax.set_title(f'{suffix}')

    # Set custom labels for x-axis every 5 positions (for zooming-in)
    positions = list(range(0, len(mutation_types), 5))
    ax.set_xticks(positions)
    ax.set_xticklabels(positions)

    # Add custom legend outside the plot area
    legend_labels = ['Deletion', 'Insertion', 'Substitution', 'Unchanged']  # Match stack order
    ax.legend(stacks, legend_labels, loc='upper left')

    # Limit the x-axis between 555 and 595 (for zooming-in)
    ax.set_xlim(555, 595)

    # Save the figure
    output_file = os.path.join(save_path, f"Mutation landscape {suffix}.png")
    plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight')
    plt.show()


# #### Mutation landscape of viable sequences

# In[14]:


# Load the CSV file into a df
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_csv/ChangesMatrix_viable_set.csv'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ChangesMatrix_viable_set.csv'
    
viable_chMatrix = pd.read_csv(file_path)

# Add a fifth row with the total sum per column
viable_chMatrix.loc['Total'] = viable_chMatrix.sum()
    
#Saving paths for the figures
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_fig'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis'

# Application of 'plot_mutation_type_distribution' function to Viable sequences set
plot_mutation_type_distribution(viable_chMatrix, 'Viable sequences')


# #### Mutation landscape of Nonviable sequences

# In[15]:


# Load the CSV file into a df
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_csv/ChangesMatrix_nonviable_set.csv'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ChangesMatrix_nonviable_set.csv'

nonviable_chMatrix = pd.read_csv(file_path)

# Add a fifth row with the total sum per column
nonviable_chMatrix.loc['Total'] = nonviable_chMatrix.sum()

#Saving paths for the figures
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_fig'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis'

# Application of 'plot_mutation_type_distribution' function to Nonviable sequences set
plot_mutation_type_distribution(nonviable_chMatrix, 'Non-viable sequences')


# #### Mutation landscape of NonML-desinged sequences

# In[16]:


# Load the CSV file into a df
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_csv/ChangesMatrix_nonml_set.csv'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ChangesMatrix_nonml_set.csv'

nonml_chMatrix = pd.read_csv(file_path)

# Add a fifth row with the total sum per column
nonml_chMatrix.loc['Total'] = nonml_chMatrix.sum()

#Saving paths for the figures
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_fig'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis'

# Application of 'plot_mutation_type_distribution' function to Nonviable sequences set
plot_mutation_type_distribution(nonml_chMatrix, 'Non-ML designed sequences')


# #### Mutation landscape of ML-desinged sequences

# In[17]:


# Load the CSV file into a df
if CLUSTER_RUN:
    file_path = r'/home/afarodrigues/embCOMP/outputs_csv/ChangesMatrix_ml_set.csv'
else:
    file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis\ChangesMatrix_ml_set.csv'
    
ml_chMatrix = pd.read_csv(file_path)

# Add a fifth row with the total sum per column
ml_chMatrix.loc['Total'] = ml_chMatrix.sum()

#Saving paths for the figures
if CLUSTER_RUN:
    save_path = r'/home/afarodrigues/embCOMP/outputs_fig'
else:
    save_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\02_Preliminary_analysis'

# Application of 'plot_mutation_type_distribution' function to Nonviable sequences set
plot_mutation_type_distribution(ml_chMatrix, 'ML designed sequences')


# In[ ]:




