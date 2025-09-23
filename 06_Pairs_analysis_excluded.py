#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import pandas as pd
import numpy as np
import os
import joblib 

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from Bio import pairwise2
from Bio.pairwise2 import format_alignment


# ### Load the models (models_1)

# In[2]:


# Directory
model_dir = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\09_Metrics_and_predictions_analysis\Models_1'
models = {}

for filename in os.listdir(model_dir):
    if filename.endswith('.pkl') and '_1' in filename:
        model_name = filename.replace('_model_1.pkl', '')
        model_path = os.path.join(model_dir, filename)

        try:
            model = joblib.load(model_path)
            models[model_name] = model

            # Dynamically assign to a variable in the global namespace
            globals()[model_name] = model

            print(f"Loaded: {model_name}")
        except Exception as e:
            print(f"Failed to load {filename}: {type(e).__name__} - {e}")


# ### Load test set data (split _1)

# In[3]:


file_path = r"C:\Users\rodri\OneDrive\Desktop\embCOMP\09_Metrics_and_predictions_analysis\Test_sets_1\final_REPs_test_1.parquet"
test_1 = pd.read_parquet(file_path)
test_1


# In[4]:


# Create a new DataFrame with the first 5 columns
prediction_split_1 = test_1.iloc[:, :5].copy()


# In[5]:


# Convert the [representation] column (which contains array-like data) into 2D numpy arrays
X_test_OHE_SVD                = np.vstack(test_1['OHE-SVD'].values)
X_test_cls_raw_EMB            = np.vstack(test_1['cls_raw_EMB'].values)
X_test_cls_t_EMB              = np.vstack(test_1['cls_t_EMB'].values)
X_test_aa_EMB                 = np.vstack(test_1['aa_EMB'].values)
X_test_aa_EMB_SVD             = np.vstack(test_1['aa_EMB_SVD'].values)
X_test_cls_raw_EMB_SVD        = np.vstack(test_1['cls_raw_EMB_SVD'].values)
X_test_cls_t_EMB_SVD          = np.vstack(test_1['cls_t_EMB_SVD'].values)
#X_test_cls_aa_EMB_concat_SVD  = np.vstack(test_1['cls_aa_EMB_concat_SVD'].values)


# In[6]:


# Make predictions using logistic regression models
preds_logR_OHE                    = logR_OHE.predict(X_test_OHE_SVD)
preds_logR_cls_raw_EMB           = logR_cls_raw_EMB.predict(X_test_cls_raw_EMB)
preds_logR_cls_t_EMB             = logR_cls_t_EMB.predict(X_test_cls_t_EMB)
preds_logR_aa_EMB                = logR_aa_EMB.predict(X_test_aa_EMB)
preds_logR_aa_EMB_SVD            = logR_aa_EMB_SVD.predict(X_test_aa_EMB_SVD)
preds_logR_cls_raw_EMB_SVD       = logR_cls_raw_EMB_SVD.predict(X_test_cls_raw_EMB_SVD)
preds_logR_cls_t_EMB_SVD         = logR_cls_t_EMB_SVD.predict(X_test_cls_t_EMB_SVD)
#preds_logR_cls_aa_EMB_concat_SVD = logR_cls_aa_EMB_concat_SVD.predict(X_test_cls_aa_EMB_concat_SVD)

# Add predictions to the DataFrame
prediction_split_1['logR_OHE_SVD']                = preds_logR_OHE
prediction_split_1['logR_cls_raw_EMB']            = preds_logR_cls_raw_EMB
prediction_split_1['logR_cls_t_EMB']              = preds_logR_cls_t_EMB
prediction_split_1['logR_aa_EMB']                 = preds_logR_aa_EMB
prediction_split_1['logR_aa_EMB_SVD']             = preds_logR_aa_EMB_SVD
prediction_split_1['logR_cls_raw_EMB_SVD']        = preds_logR_cls_raw_EMB_SVD
prediction_split_1['logR_cls_t_EMB_SVD']          = preds_logR_cls_t_EMB_SVD
#prediction_split_1['logR_cls_aa_EMB_concat_SVD']  = preds_logR_cls_aa_EMB_concat_SVD

prediction_split_1


# In[7]:


# Random Forests

# Make predictions using the models
preds_rfo_OHE                   = rfo_OHE.predict(X_test_OHE_SVD)
preds_rfo_cls_raw_EMB          = rfo_cls_raw_EMB.predict(X_test_cls_raw_EMB)
preds_rfo_cls_t_EMB            = rfo_cls_t_EMB.predict(X_test_cls_t_EMB)
preds_rfo_aa_EMB               = rfo_aa_EMB.predict(X_test_aa_EMB)
preds_rfo_aa_EMB_SVD           = rfo_aa_EMB_SVD.predict(X_test_aa_EMB_SVD)
preds_rfo_cls_raw_EMB_SVD      = rfo_cls_raw_EMB_SVD.predict(X_test_cls_raw_EMB_SVD)
preds_rfo_cls_t_EMB_SVD        = rfo_cls_t_EMB_SVD.predict(X_test_cls_t_EMB_SVD)
#preds_rfo_cls_aa_EMB_concat_SVD = rfo_cls_aa_EMB_concat_SVD.predict(X_test_cls_aa_EMB_concat_SVD)

# Add predictions to the DataFrame
prediction_split_1['rfo_OHE_SVD']                = preds_rfo_OHE
prediction_split_1['rfo_cls_raw_EMB']            = preds_rfo_cls_raw_EMB
prediction_split_1['rfo_cls_t_EMB']              = preds_rfo_cls_t_EMB
prediction_split_1['rfo_aa_EMB']                 = preds_rfo_aa_EMB
prediction_split_1['rfo_aa_EMB_SVD']             = preds_rfo_aa_EMB_SVD
prediction_split_1['rfo_cls_raw_EMB_SVD']        = preds_rfo_cls_raw_EMB_SVD
prediction_split_1['rfo_cls_t_EMB_SVD']          = preds_rfo_cls_t_EMB_SVD
#prediction_split_1['rfo_cls_aa_EMB_concat_SVD']  = preds_rfo_cls_aa_EMB_concat_SVD

prediction_split_1


# In[8]:


# MLP

# Make predictions using the models
preds_mlp_OHE                   = mlp_OHE.predict(X_test_OHE_SVD)
preds_mlp_cls_raw_EMB          = mlp_cls_raw_EMB.predict(X_test_cls_raw_EMB)
preds_mlp_cls_t_EMB            = mlp_cls_t_EMB.predict(X_test_cls_t_EMB)
preds_mlp_aa_EMB               = mlp_aa_EMB.predict(X_test_aa_EMB)
preds_mlp_aa_EMB_SVD           = mlp_aa_EMB_SVD.predict(X_test_aa_EMB_SVD)
preds_mlp_cls_raw_EMB_SVD      = mlp_cls_raw_EMB_SVD.predict(X_test_cls_raw_EMB_SVD)
preds_mlp_cls_t_EMB_SVD        = mlp_cls_t_EMB_SVD.predict(X_test_cls_t_EMB_SVD)
#preds_mlp_cls_aa_EMB_concat_SVD = mlp_cls_aa_EMB_concat_SVD.predict(X_test_cls_aa_EMB_concat_SVD)

# Add predictions to the DataFrame
prediction_split_1['mlp_OHE_SVD']                = preds_mlp_OHE
prediction_split_1['mlp_cls_raw_EMB']            = preds_mlp_cls_raw_EMB
prediction_split_1['mlp_cls_t_EMB']              = preds_mlp_cls_t_EMB
prediction_split_1['mlp_aa_EMB']                 = preds_mlp_aa_EMB
prediction_split_1['mlp_aa_EMB_SVD']             = preds_mlp_aa_EMB_SVD
prediction_split_1['mlp_cls_raw_EMB_SVD']        = preds_mlp_cls_raw_EMB_SVD
prediction_split_1['mlp_cls_t_EMB_SVD']          = preds_mlp_cls_t_EMB_SVD
#prediction_split_1['mlp_cls_aa_EMB_concat_SVD']  = preds_mlp_cls_aa_EMB_concat_SVD


# In[9]:


# Create the base of the new df with the first 5 columns
correctness = prediction_split_1.iloc[:, :5].copy()

# Compare each prediction to ground truth and assign 'correct' or 'incorrect'
for col in prediction_split_1.columns[5:]:
    correctness[col] = prediction_split_1.apply(
        lambda row: 'correct' if row[col] == row['Viability'] else 'incorrect',
        axis=1
    )

correctness


# ### Get 'functional' groups

# In[10]:


# Select prediction columns only (all columns except the first 5 metadata columns)
prediction_cols = correctness.columns[5:]

# Filter rows where all predictions are 'correct'
easy_sequences = correctness[
    correctness[prediction_cols].apply(lambda row: all(x == 'correct' for x in row), axis=1)
].copy()

easy_sequences


# In[11]:


## Difficult sequences

# Select prediction columns only
prediction_cols = correctness.columns[5:]

# Filter rows where all predictions are 'incorrect'
difficult_sequences = correctness[
    correctness[prediction_cols].apply(lambda row: all(x == 'incorrect' for x in row), axis=1)
].copy()

difficult_sequences


# In[12]:


## OHE incorrect sequences

# Select columns containing 'OHE' in their names (prediction columns only)
ohe_cols = [col for col in correctness.columns[5:] if 'OHE' in col]

# Filter rows where all 'OHE' prediction is 'incorrect'
ohe_incorrect_sequences = correctness[
    correctness[ohe_cols].apply(lambda row: all(x == 'incorrect' for x in row), axis=1)
].copy()

ohe_incorrect_sequences


# In[13]:


## cls_raw_EMB incorrect sequences

# Select columns containing 'cls_raw_EMB' in their names (prediction columns only)
cls_raw_EMB_cols = [col for col in correctness.columns[5:] if 'cls_raw_EMB' in col]

# Filter rows where all 'cls_raw_EMB' prediction is 'incorrect'
cls_raw_EMB_incorrect_sequences = correctness[
    correctness[cls_raw_EMB_cols].apply(lambda row: all(x == 'incorrect' for x in row), axis=1)
].copy()

cls_raw_EMB_incorrect_sequences


# In[14]:


## cls_t_EMB incorrect sequences

# Select columns containing 'cls_t_EMB' in their names (prediction columns only)
cls_t_EMB_cols = [col for col in correctness.columns[5:] if 'cls_t_EMB' in col]

# Filter rows where all 'cls_t_EMB' prediction is 'incorrect'
cls_t_EMB_incorrect_sequences = correctness[
    correctness[cls_t_EMB_cols].apply(lambda row: all(x == 'incorrect' for x in row), axis=1)
].copy()

cls_t_EMB_incorrect_sequences


# In[15]:


## aa_EMB incorrect sequences

# Select columns containing 'aa_EMB' in their names (prediction columns only)
aa_EMB_cols = [col for col in correctness.columns[5:] if 'aa_EMB' in col]

# Filter rows where all 'cls_aa_EMB' prediction is 'incorrect'
aa_EMB_incorrect_sequences = correctness[
    correctness[aa_EMB_cols].apply(lambda row: all(x == 'incorrect' for x in row), axis=1)
].copy()

aa_EMB_incorrect_sequences


# ## Plotting

# In[16]:


# Sample 50 random rows (without replacement)
sampled_correctness = correctness.sample(n=50, random_state=42)

# Convert 'correct'/'incorrect' to 1/0
correctness_numeric = sampled_correctness.iloc[:, 5:].applymap(lambda x: 1 if x == 'correct' else 0)

# Sort sampled rows by total correct predictions descending
sorted_correctness = correctness_numeric.loc[correctness_numeric.sum(axis=1).sort_values(ascending=False).index]

# Keep sampled indices for y-axis labels
original_indices = sorted_correctness.index

# Define colors for representation types
rep_color_dict = {
    'OHE': '#2ca02c',                  # Green
    'cls_raw_EMB': '#1f77b4',         # Blue
    'cls_raw_EMB_SVD': '#1f77b4',     # Blue (same as cls_raw_EMB)
    'cls_t_EMB': '#ff7f0e',           # Orange
    'cls_t_EMB_SVD': '#ff7f0e',       # Orange (same as cls_t_EMB)
    'aa_EMB': '#d62728',              # Red
    'aa_EMB_SVD': '#d62728'          # Red (same as aa_EMB)
}

# Build color dict for columns based on substring in column names
color_dict = {}
for col in sorted_correctness.columns:
    for rep, color in rep_color_dict.items():
        if rep in col:
            color_dict[col] = color
            break

# Plot heatmap
plt.figure(figsize=(15, 8))

for idx, column in enumerate(sorted_correctness.columns):
    heatmap_data = np.full_like(sorted_correctness, np.nan, dtype=np.float32)
    heatmap_data[:, idx] = sorted_correctness[column]

    cmap = ListedColormap(['white', color_dict.get(column, '#cccccc')])

    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        cbar=False,
        annot=False,
        fmt="d",
        linewidths=0.5,
        linecolor='black',
        vmin=0, vmax=1
    )

# Draw border
ax = plt.gca()
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1)

# Title and labels
plt.title("Correctness across model-representation pairs in test sequences (sampled 50)", fontsize=12)
plt.xlabel("Model and representation type", labelpad=10, fontsize=10)
plt.ylabel("Sequence index", fontsize=10)

# Ticks
plt.xticks(ticks=np.arange(len(sorted_correctness.columns)) + 0.5,
           labels=sorted_correctness.columns, rotation=45, ha='right', fontsize=8)
plt.yticks(ticks=np.arange(len(original_indices)) + 0.5,
           labels=original_indices, rotation=0, fontsize=6)

# Legend
legend_elements = [Patch(facecolor=color, edgecolor='black', label=rep) for rep, color in rep_color_dict.items()]
plt.legend(handles=legend_elements,
           title='Representation type',
           bbox_to_anchor=(1.02, 1),
           loc='upper left',
           borderaxespad=0.,
           fontsize=12,
           title_fontsize=14)

# Save and show
plt.savefig('Correctness_heatmap_sampled_50.png', dpi=600, bbox_inches='tight')
plt.show()


# In[17]:


# Compute normalized value counts for each group
easy_counts = easy_sequences['Viability'].value_counts(normalize=True) * 100
difficult_counts = difficult_sequences['Viability'].value_counts(normalize=True) * 100
missed_ohe_counts = ohe_incorrect_sequences['Viability'].value_counts(normalize=True) * 100
missed_cls_raw_EMB_counts = cls_raw_EMB_incorrect_sequences['Viability'].value_counts(normalize=True) * 100
missed_cls_t_EMB_counts = cls_t_EMB_incorrect_sequences['Viability'].value_counts(normalize=True) * 100
missed_aa_EMB_counts = aa_EMB_incorrect_sequences['Viability'].value_counts(normalize=True) * 100

# Map numeric viability labels to strings
label_mapping = {0: 'Non-viable', 1: 'Viable'}
easy_counts.index = easy_counts.index.map(label_mapping)
difficult_counts.index = difficult_counts.index.map(label_mapping)
missed_ohe_counts.index = missed_ohe_counts.index.map(label_mapping)
missed_cls_raw_EMB_counts.index = missed_cls_raw_EMB_counts.index.map(label_mapping)
missed_cls_t_EMB_counts.index = missed_cls_t_EMB_counts.index.map(label_mapping)
missed_aa_EMB_counts.index = missed_aa_EMB_counts.index.map(label_mapping)

# Define consistent order and fill missing categories with 0
categories = ['Non-viable', 'Viable']
easy_counts = easy_counts.reindex(categories).fillna(0)
difficult_counts = difficult_counts.reindex(categories).fillna(0)
missed_ohe_counts = missed_ohe_counts.reindex(categories).fillna(0)
missed_cls_raw_EMB_counts = missed_cls_raw_EMB_counts.reindex(categories).fillna(0)
missed_cls_t_EMB_counts = missed_cls_t_EMB_counts.reindex(categories).fillna(0)
missed_aa_EMB_counts = missed_aa_EMB_counts.reindex(categories).fillna(0)

# Create DataFrame for plotting
composition_df = pd.DataFrame({
    'Easy sequences': easy_counts,
    'Difficult sequences': difficult_counts,
    'Missed by OHE': missed_ohe_counts,
    'Missed by CLS_raw_EMB': missed_cls_raw_EMB_counts,
    'Missed by CLS_t_EMB': missed_cls_t_EMB_counts,
    'Missed by AA_EMB': missed_aa_EMB_counts
}).T

# Plotting
fig, ax = plt.subplots(figsize=(7, 6))

composition_df.plot(
    kind='bar',
    stacked=True,
    color=['black', 'orange'],
    edgecolor='black',
    ax=ax
)

ax.set_ylabel('Proportion among total sequences (%)')
ax.set_title('Viability Composition')

ax.legend(
    title='Viability',
    title_fontsize=14,
    fontsize=12,
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.xticks(rotation=45, ha='right')

plt.savefig('Viability_composition_groups_excluded.png', dpi=600, bbox_inches='tight')
plt.show()


# In[18]:


# Define model groups
model_groups = {
    'ML': ['cnn_designed_plus_rand_train_walked', 'cnn_rand_doubles_plus_single_seed',
           'cnn_standard_walked', 'cnn_rand_doubles_plus_single_walked',
           'cnn_designed_plus_rand_train_seed', 'cnn_standard_seed', 'rnn_rand_doubles_plus_singles_seed',
           'rnn_designed_plus_rand_train_walked', 'rnn_standard_seed', 'rnn_standard_walked',
           'rnn_designed_plus_rand_train_seed', 'rnn_rand_doubles_plus_singles_walked',
           'lr_rand_doubles_plus_single_walked', 'lr_standard_walked',
           'lr_rand_doubles_plus_single_seed', 'lr_designed_plus_rand_train_seed',
           'lr_standard_seed', 'lr_designed_plus_rand_train_walked'],
    'non-ML': ['designed', 'rand', 'random_doubles', 'single']
}

# Function to classify each sequence
def map_to_model(subset):
    for model, categories in model_groups.items():
        if subset in categories:
            return model
    return 'Other'

# Group all DataFrames
all_dfs = {
    'Easy sequences': easy_sequences.copy(),
    'Difficult sequences': difficult_sequences.copy(),
    'Missed by OHE': ohe_incorrect_sequences.copy(),
    'Missed by CLS_raw_EMB': cls_raw_EMB_incorrect_sequences.copy(),
    'Missed by CLS_t_EMB': cls_t_EMB_incorrect_sequences.copy(),
    'Missed by AA_EMB': aa_EMB_incorrect_sequences.copy()
}

# Compute model composition
composition_model_data = {}
for name, df in all_dfs.items():
    df['Model'] = df['Subset'].apply(map_to_model)
    model_counts = df['Model'].value_counts(normalize=True) * 100
    model_counts = model_counts.reindex(['ML', 'non-ML'], fill_value=0)
    composition_model_data[name] = model_counts

# DataFrame for plotting
composition_model_df = pd.DataFrame(composition_model_data).T

# Plotting
fig, ax = plt.subplots(figsize=(7, 6))

composition_model_df.plot(
    kind='bar',
    stacked=True,
    color=['blue', 'red'],
    edgecolor='black',
    ax=ax
)

# Labels and title
ax.set_ylabel('Proportion among total sequences (%)')
ax.set_title('Design Strategy Composition Across Groups')

# Custom legend
custom_labels = ['ML-designed', 'Non-ML-designed']
ax.legend(
    title='Design strategy',
    labels=custom_labels,
    title_fontsize=14,
    fontsize=12,
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

# Save and display
plt.savefig('Design_strategy_composition_all_groups_excluded.png', dpi=600, bbox_inches='tight')
plt.show()


# In[19]:


stop


# ## Mutation landscape analysis

# In[ ]:


### Function to plot mutation type across positions
def plot_mutation_type_distribution(df, suffix):
    # Calculate percentages for each mutation type at each position
    total_mutations = df.iloc[-1]  # Total mutations per column
    mutation_types = df.iloc[:-1] / total_mutations  # Divide each mutation type count by total mutations
    mutation_types = mutation_types.transpose() * 100  # Convert to percentages and transpose for plotting

    # Create a figure with custom size
    fig, ax = plt.subplots(figsize=(6, 5))  # Adjust fhe figsize parameter if not zoomed-in

    # Define custom colors for stacks
    colors = ['blue', 'green', 'red', 'white']

    # Reverse the order of mutation types for stacking
    mutation_types = mutation_types.iloc[:, ::-1]

    # Plot theme river with custom colors and reversed stacking order
    stacks = ax.stackplot(range(len(mutation_types)), mutation_types.values.T, labels=mutation_types.columns, colors=colors)
    ax.set_xlabel('Amino acid position')
    ax.set_ylabel('Percentage of change')
    ax.set_title(f'{suffix}')

    # Set custom labels for x-axis every 50 positions
    #positions = list(range(0, len(mutation_types), 100))
    
    # Set custom labels for x-axis every 5 positions (only for zooming-in)
    positions = list(range(0, len(mutation_types), 5))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(positions)

    # Add custom legend outside the plot area
    legend_labels = ['Deletion', 'Insertion', 'Substitution', 'Unchanged']  # Reverse the legend labels order
    ax.legend(stacks, legend_labels, loc='upper left')

    # Limit the x-axis between 550 and 600 (only for zooming-in)
    ax.set_xlim(555, 595)

    # Save the figure as a PNG file using the suffix as the filename
    output_file = f"Mutation landscape {suffix}.png"
    plt.savefig(output_file, format='png', bbox_inches='tight')
    
    plt.show()


# In[ ]:


## Function detect changes
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


# In[ ]:


# AAV2 VP1 reference reference
aav2vp1_refSeq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

## Function 'make_changes_matrix' to generate changes matrix
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


# In[ ]:


## Call 'make_changes_function' and apply to easy sequences
easy_matrix = make_changes_matrix(easy_sequences, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(easy_matrix)
csv_path = 'Easy_seqs_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
easy_chMatrix = pd.read_csv('Easy_seqs_changesMatrix.csv')

# Add a fifth row with the total sum per column
easy_chMatrix.loc['Total'] = easy_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to easy sequences
plot_mutation_type_distribution(easy_chMatrix, 'Easy sequences')


# In[ ]:


## Call 'make_changes_function' and apply to difficult sequences
difficult_matrix = make_changes_matrix(difficult_sequences, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(difficult_matrix)
csv_path = 'Difficult_seqs_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
difficult_chMatrix = pd.read_csv('Difficult_seqs_changesMatrix.csv')

# Add a fifth row with the total sum per column
difficult_chMatrix.loc['Total'] = difficult_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to easy sequences
plot_mutation_type_distribution(difficult_chMatrix, 'Difficult sequences')


# ### OHE consistently misclassified sequences

# In[20]:


## Split incorrect OHE sequences by viability

# DataFrame with viable sequences
ohe_incorrect_viable = ohe_incorrect_sequences[ohe_incorrect_sequences['Viability'] == 1]

# DataFrame with non-viable sequences
ohe_incorrect_non_viable = ohe_incorrect_sequences[ohe_incorrect_sequences['Viability'] == 0]


# In[22]:


ohe_incorrect_non_viable


# In[ ]:


## Call 'make_changes_function' and apply to ohe_incorrect_sequences & viable
OHE_missed_viable_matrix = make_changes_matrix(ohe_incorrect_viable, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(OHE_missed_viable_matrix)
csv_path = 'OHE_missed_viable_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
OHE_missed_viable_chMatrix = pd.read_csv('OHE_missed_viable_changesMatrix.csv')

# Add a fifth row with the total sum per column
OHE_missed_viable_chMatrix.loc['Total'] = OHE_missed_viable_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to ohe_incorrect_sequences & viable
plot_mutation_type_distribution(OHE_missed_viable_chMatrix, 'OHE missed viable sequences')


# In[ ]:


## Call 'make_changes_function' and apply to ohe_incorrect_sequences & non-viable
OHE_missed_non_viable_matrix = make_changes_matrix(ohe_incorrect_non_viable, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(OHE_missed_non_viable_matrix)
csv_path = 'OHE_missed_non_viable_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
OHE_missed_non_viable_chMatrix = pd.read_csv('OHE_missed_non_viable_changesMatrix.csv')

# Add a fifth row with the total sum per column
OHE_missed_non_viable_chMatrix.loc['Total'] = OHE_missed_non_viable_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to ohe_incorrect_sequences & non-viable
plot_mutation_type_distribution(OHE_missed_non_viable_chMatrix, 'OHE missed non-viable sequences')


# ### CLS_raw_EMB consistently misclassified sequences

# In[23]:


## Split incorrect cls_raw_EMB sequences by viability

# DataFrame with viable sequences
cls_raw_EMB_incorrect_viable = cls_raw_EMB_incorrect_sequences[cls_raw_EMB_incorrect_sequences['Viability'] == 1]

# DataFrame with non-viable sequences
cls_raw_EMB_incorrect_non_viable = cls_raw_EMB_incorrect_sequences[cls_raw_EMB_incorrect_sequences['Viability'] == 0]


# In[25]:


cls_raw_EMB_incorrect_non_viable


# In[ ]:


## Call 'make_changes_function' and apply to cls_raw_EMB_incorrect_sequences & viable
cls_raw_EMB_missed_viable_matrix = make_changes_matrix(cls_raw_EMB_incorrect_viable, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(cls_raw_EMB_missed_viable_matrix)
csv_path = 'cls_raw_EMB_missed_viable_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
cls_raw_EMB_missed_viable_chMatrix = pd.read_csv('cls_raw_EMB_missed_viable_changesMatrix.csv')

# Add a fifth row with the total sum per column
cls_raw_EMB_missed_viable_chMatrix.loc['Total'] = cls_raw_EMB_missed_viable_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to cls_raw_EMB_incorrect_sequences & viable
plot_mutation_type_distribution(cls_raw_EMB_missed_viable_chMatrix, 'cls_raw_EMB missed viable sequences')


# In[ ]:


## Call 'make_changes_function' and apply to cls_raw_EMB_incorrect_sequences & non-viable
cls_raw_EMB_missed_non_viable_matrix = make_changes_matrix(cls_raw_EMB_incorrect_non_viable, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(cls_raw_EMB_missed_non_viable_matrix)
csv_path = 'cls_raw_EMB_missed_non_viable_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
cls_raw_EMB_missed_non_viable_chMatrix = pd.read_csv('cls_raw_EMB_missed_non_viable_changesMatrix.csv')

# Add a fifth row with the total sum per column
cls_raw_EMB_missed_non_viable_chMatrix.loc['Total'] = cls_raw_EMB_missed_non_viable_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to cls_raw_EMB_incorrect_sequences & non-viable
plot_mutation_type_distribution(cls_raw_EMB_missed_non_viable_chMatrix, 'cls_raw_EMB missed non viable sequences')


# ### CLS_t_EMB consistently misclassified sequences

# In[26]:


## Split incorrect cls_t_EMB sequences by viability

# DataFrame with viable sequences
cls_t_EMB_incorrect_viable = cls_t_EMB_incorrect_sequences[cls_t_EMB_incorrect_sequences['Viability'] == 1]

# DataFrame with non-viable sequences
cls_t_EMB_incorrect_non_viable = cls_t_EMB_incorrect_sequences[cls_t_EMB_incorrect_sequences['Viability'] == 0]


# In[28]:


cls_t_EMB_incorrect_non_viable


# In[ ]:


## Call 'make_changes_function' and apply to cls_t_EMB_incorrect_sequences & viable
cls_t_EMB_missed_viable_matrix = make_changes_matrix(cls_t_EMB_incorrect_viable, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(cls_t_EMB_missed_viable_matrix)
csv_path = 'cls_t_EMB_missed_viable_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


## Call 'make_changes_function' and apply to cls_t_EMB_incorrect_sequences & non-viable
cls_t_EMB_missed_non_viable_matrix = make_changes_matrix(cls_t_EMB_incorrect_non_viable, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(cls_t_EMB_missed_non_viable_matrix)
csv_path = 'cls_t_EMB_missed_non_viable_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
cls_t_EMB_missed_non_viable_chMatrix = pd.read_csv('cls_t_EMB_missed_non_viable_changesMatrix.csv')

# Add a fifth row with the total sum per column
cls_t_EMB_missed_non_viable_chMatrix.loc['Total'] = cls_t_EMB_missed_non_viable_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to cls_t_EMB_incorrect_sequences & non-viable
plot_mutation_type_distribution(cls_t_EMB_missed_non_viable_chMatrix, 'cls_t_EMB missed non viable sequences')


# ### AA_EMB consistently misclassified sequences

# In[29]:


## Split incorrect aa_EMB sequences by viability

# DataFrame with viable sequences
aa_EMB_incorrect_viable = aa_EMB_incorrect_sequences[aa_EMB_incorrect_sequences['Viability'] == 1]

# DataFrame with non-viable sequences
aa_EMB_incorrect_non_viable = aa_EMB_incorrect_sequences[aa_EMB_incorrect_sequences['Viability'] == 0]


# In[31]:


aa_EMB_incorrect_non_viable


# In[ ]:


## Call 'make_changes_function' and apply to aa_EMB_incorrect_sequences & viable
aa_EMB_missed_viable_matrix = make_changes_matrix(aa_EMB_incorrect_viable, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(aa_EMB_missed_viable_matrix)
csv_path = 'aa_EMB_missed_viable_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
aa_EMB_missed_viable_chMatrix = pd.read_csv('aa_EMB_missed_viable_changesMatrix.csv')

# Add a fifth row with the total sum per column
aa_EMB_missed_viable_chMatrix.loc['Total'] = aa_EMB_missed_viable_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to aa_EMB_incorrect_sequences & non-viable
plot_mutation_type_distribution(aa_EMB_missed_viable_chMatrix, 'aa_EMB missed viable sequences')


# In[ ]:


## Call 'make_changes_function' and apply to aa_EMB_incorrect_sequences & non-viable
aa_EMB_missed_non_viable_matrix = make_changes_matrix(aa_EMB_incorrect_non_viable, refSeq=aav2vp1_refSeq)

# Save the dfto a CSV file
df_result = pd.DataFrame(aa_EMB_missed_non_viable_matrix)
csv_path = 'aa_EMB_missed_non_viable_changesMatrix.csv'
df_result.to_csv(csv_path, index=False)

# Print the columns including the 561-588 (fragment) region
print("Result matrix (columns 557 to 590):")
print(df_result.iloc[:, 558:590])  


# In[ ]:


# Load the CSV file into a df
aa_EMB_missed_non_viable_chMatrix = pd.read_csv('aa_EMB_missed_non_viable_changesMatrix.csv')

# Add a fifth row with the total sum per column
aa_EMB_missed_non_viable_chMatrix.loc['Total'] = aa_EMB_missed_non_viable_chMatrix.sum()

# Application of 'plot_mutation_type_distribution' function to aa_EMB_incorrect_sequences & non-viable
plot_mutation_type_distribution(aa_EMB_missed_non_viable_chMatrix, 'aa_EMB missed non viable sequences')


# In[ ]:




