#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


# In[2]:


# Load df with non-fine-tuned embeddings
file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\03_Encodings_generation\final_REPs_train.parquet'
final_REPs_train = pd.read_parquet(file_path)

#Inspect
print(final_REPs_train)


# ### Load fine-tuned embeddings

# In[3]:


## For CLS_raw_EMB

# Read file
with open('embeddings_raw_106_combined.pkl', 'rb') as f:
    data = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Clean up the sequence column
df['sequence'] = df['sequence'].apply(lambda x: x.replace('[CLS]', '')
                                                   .replace('[SEP]', '')
                                                   .strip()
                                                   .replace(' ', ''))
print(df)


# In[4]:


## For CLS_raw_EMB
# Check for matches between non-fine-tuned embeddings df and fine-tuned embeddings df
matched = df['sequence'].isin(final_REPs_train['Sequence'])

# Add a column to show if a match was found
df['in_parquet'] = matched

# Summary
print(f"{matched.sum()} out of {len(df)} sequences found in the parquet file.")


# In[5]:


## For CLS_raw_EMB
# Create a mapping from 'sequence' in df to 'Sequence' in final_REPs_train and transfer fine-tuned embedding from df to final_REPs_train
seq_to_embedding = df.set_index('sequence')['embedding'].to_dict()

# Map embeddings into final_REPs_train df
final_REPs_train['cls_raw_EMB_FT'] = final_REPs_train['Sequence'].map(seq_to_embedding)
final_REPs_train


# In[6]:


## For CLS_t_EMB
# Read file
with open('embeddings_pooler_106_combined.pkl', 'rb') as f:
    data = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Clean up the sequence column
df['sequence'] = df['sequence'].apply(lambda x: x.replace('[CLS]', '')
                                                   .replace('[SEP]', '')
                                                   .strip()
                                                  .replace(' ', ''))
#Inspect
print(df)


# In[7]:


## For CLS_t_EMB
# Check for matches between non-fine-tuned embeddings df and fine-tuned embeddings df
matched = df['sequence'].isin(final_REPs_train['Sequence'])

# Add a column to show if a match was found
df['in_parquet'] = matched

# Summary
print(f"{matched.sum()} out of {len(df)} sequences found in the parquet file.")


# In[8]:


## For CLS_t_EMB
# Create a mapping from 'sequence' in df to 'Sequence' in final_REPs_train and transfer fine-tuned embedding from df to final_REPs_train
seq_to_embedding = df.set_index('sequence')['embedding'].to_dict()

# Map embeddings into df2
final_REPs_train['cls_t_EMB_FT'] = final_REPs_train['Sequence'].map(seq_to_embedding)

#Inspect
print(final_REPs_train)


# In[9]:


## For AA_EMB
# Read file
with open('embeddings_mean_106_combined.pkl', 'rb') as f:
    data = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Clean up the sequence column
df['sequence'] = df['sequence'].apply(lambda x: x.replace('[CLS]', '')
                                                   .replace('[SEP]', '')
                                                   .strip()
                                                   .replace(' ', ''))
#Inspect
print(df)


# In[10]:


## For AA_EMB
# Check for matches between non-fine-tuned embeddings df and fine-tuned embeddings df
matched = df['sequence'].isin(final_REPs_train['Sequence'])

# Add a column to show if a match was found
df['in_parquet'] = matched

# Summary
print(f"{matched.sum()} out of {len(df)} sequences found in the parquet file.")


# In[11]:


## For AA_EMB

# Create a mapping from sequence in df to Sequence in final_REPs_train and transfering to embedding from df to final_REPs_train
seq_to_embedding = df.set_index('sequence')['embedding'].to_dict()

# Map embeddings into df2
final_REPs_train['aa_EMB_FT'] = final_REPs_train['Sequence'].map(seq_to_embedding)

# Inspect
print(final_REPs_train)


# ### t-SNE of fine-tuned embeddings by labels

# In[12]:


# Function 'tsne_and_visualize_by_viability'
def tsne_and_visualize_by_viability(result_df_encoded, features_column, perplexity=50, max_iter=10000, random_state=42, save_path=None, title_suffix=None):
    """
    Performs t-SNE dimensionality reduction on the given features and visualizes the result,
    coloring points by categorical viability (0 = Non-viable, 1 = Viable).
    """
    if title_suffix is None:
        title_suffix = features_column

    # Extract feature data
    encoded_sequences = np.vstack(result_df_encoded[features_column].values)

    # Initialize and fit t-SNE
    tsne = TSNE(perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    tsne_data = tsne.fit_transform(encoded_sequences)

    # Add t-SNE coordinates to DataFrame
    result_df_encoded['tsne_x'] = tsne_data[:, 0]
    result_df_encoded['tsne_y'] = tsne_data[:, 1]

    # Map viability labels
    result_df_encoded['Viability_label'] = result_df_encoded['Viability'].map({0: 'Non-viable', 1: 'Viable'})

    # Define custom colors
    viability_colors = {'Non-viable': 'black', 'Viable': 'orange'}

    # Plot
    plt.figure(figsize=(9.3, 6))
    sns.scatterplot(
        x='tsne_x', y='tsne_y',
        hue='Viability_label',
        palette=viability_colors,
        data=result_df_encoded,
        alpha=0.8, s=20
    )

    # Titles and labels
    plt.title(f't-SNE of {title_suffix} colored by Viability')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # Legend positioning
    plt.legend(
        title='Viability',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )

    # Adjust layout for external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save path logic
    save_dir = os.path.normpath(r'C:\Users\rodri\OneDrive\Desktop\embCOMP\11_Fine-tunning')
    filename = f't-SNE_{title_suffix}_by_viability.png'
    save_path = os.path.join(save_dir, filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.show()


# In[13]:


def tsne_and_visualize_by_design(
    result_df_encoded, 
    features_column, 
    perplexity=50, 
    max_iter=10000, 
    random_state=42, 
    save_path=None, 
    title_suffix=None
):
    """
    Performs t-SNE on the given feature column and visualizes model design strategies,
    grouping ML-designed sequences by model type (CNN, RNN, LR) and comparing with non-ML.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    from sklearn.manifold import TSNE

    if title_suffix is None:
        title_suffix = features_column

    # Extract feature vectors
    encoded_sequences = np.vstack(result_df_encoded[features_column].values)

    # Run t-SNE
    tsne = TSNE(perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    tsne_data = tsne.fit_transform(encoded_sequences)

    # Add t-SNE coordinates
    result_df_encoded['tsne_x'] = tsne_data[:, 0]
    result_df_encoded['tsne_y'] = tsne_data[:, 1]

    # Define non-ML subsets
    non_ml_subsets = ['designed', 'rand', 'random_doubles', 'single']

    # Classify design strategy
    def classify_strategy(subset):
        if subset in non_ml_subsets:
            return 'Non-ML'
        elif subset.startswith('cnn'):
            return 'CNN'
        elif subset.startswith('rnn'):
            return 'RNN'
        elif subset.startswith('lr'):
            return 'LR'

    result_df_encoded['Design_Group'] = result_df_encoded['Subset'].apply(classify_strategy)

    # Map to more descriptive labels for plotting
    label_map = {
        'Non-ML': 'Non-ML designed',
        'CNN': 'CNN-based design',
        'RNN': 'RNN-based design',
        'LR': 'LogR-based design'
    }
    result_df_encoded['Design_Group_Pretty'] = result_df_encoded['Design_Group'].map(label_map)

    # Define color palette matching the pretty labels
    group_palette = {
        'Non-ML designed': 'blue',
        'CNN-based design': 'red',
        'RNN-based design': 'green',
        'LogR-based design': 'orange'
    }

    # Plot each group separately to control layering order
    plt.figure(figsize=(10, 6))

    # Order to plot (Non-ML last to be on top)
    order = [
        'CNN-based design',
        'RNN-based design',
        'LogR-based design',
        'Non-ML designed'
    ]

    for group in order:
        subset = result_df_encoded[result_df_encoded['Design_Group_Pretty'] == group]
        if not subset.empty:
            sns.scatterplot(
                x='tsne_x', y='tsne_y',
                data=subset,
                color=group_palette[group],
                label=group,
                alpha=0.7, s=20, linewidth=0
            )

    plt.title(f't-SNE of {title_suffix} by design strategy')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(
        title='Design Group',
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save path logic
    save_dir = os.path.normpath(r'C:\Users\rodri\OneDrive\Desktop\embCOMP\11_Fine-tunning')
    filename = f't-SNE_{title_suffix}_by_strategy.png'
    save_path = os.path.join(save_dir, filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.show()


# In[14]:


# t-SNE for cls_raw_EMB_FT by viability
tsne_and_visualize_by_viability(
    result_df_encoded=final_REPs_train.sample(n=10000, random_state=42),
    features_column='cls_raw_EMB_FT' ,
    perplexity=50, 
    max_iter=10000,
    random_state=42,
    title_suffix= 'cls_raw_EMB_FT',
)


# In[15]:


# t-SNE for cls_raw_EMB_FT by viability
tsne_and_visualize_by_design(
    result_df_encoded=final_REPs_train.sample(n=10000, random_state=42),
    features_column='cls_raw_EMB_FT' ,
    perplexity=50, 
    max_iter=10000,
    random_state=42,
    title_suffix= 'cls_raw_EMB_FT',
)


# In[16]:


# t-SNE for cls_t_EMB_FT by viability
tsne_and_visualize_by_viability(
    result_df_encoded=final_REPs_train.sample(n=10000, random_state=42),
    features_column='cls_t_EMB_FT' ,
    perplexity=50, 
    max_iter=10000,
    random_state=42,
    title_suffix= 'cls_t_EMB_FT',
)


# In[17]:


# t-SNE for cls_t_EMB_FT by viability
tsne_and_visualize_by_design(
    result_df_encoded=final_REPs_train.sample(n=10000, random_state=42),
    features_column='cls_t_EMB_FT' ,
    perplexity=50, 
    max_iter=10000,
    random_state=42,
    title_suffix= 'cls_t_EMB_FT',
)


# In[18]:


# t-SNE for aa_EMB_FT by viability
tsne_and_visualize_by_viability(
    result_df_encoded=final_REPs_train.sample(n=10000, random_state=42),
    features_column='aa_EMB_FT' ,
    perplexity=50, 
    max_iter=10000,
    random_state=42,
    title_suffix= 'aa_EMB_FT',
)


# In[19]:


# t-SNE for aa_EMB_FT by viability
tsne_and_visualize_by_design(
    result_df_encoded=final_REPs_train.sample(n=10000, random_state=42),
    features_column='aa_EMB_FT' ,
    perplexity=50, 
    max_iter=10000,
    random_state=42,
    title_suffix= 'aa_EMB_FT',
)


# ### Magnitude of change for embeddings before and after fine-tuning

# In[20]:


## Function 'add_embedding_difference'
def add_embedding_absolute_percentage_difference_vec(df, col_a, col_b, new_col_name=None):
    """
    Computes the absolute percentage difference between two columns of embeddings in a DataFrame.

    This vectorized function calculates the element-wise absolute percentage difference between
    embeddings stored as lists or arrays in `col_a` and `col_b`. The result for each row is
    stored as a list in a new column. Division by zero is handled such that any resulting
    NaN values are set to zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the embedding columns.
    col_a : str
        Name of the first column of embeddings (numerator for percentage difference).
    col_b : str
        Name of the second column of embeddings (denominator for percentage difference).
    new_col_name : str, optional
        Name of the new column to store the absolute percentage differences. If None, a default
        name is generated as "{col_a}_abs_pct_minus_{col_b}".

    Returns
    -------
    pandas.DataFrame
        DataFrame with the new column containing the absolute percentage differences between
        the embeddings from `col_a` and `col_b`.
    """
    if new_col_name is None:
        new_col_name = f"{col_a}_abs_pct_minus_{col_b}"

    # Convert the entire column of lists/arrays into a 2D array
    arr_a = np.stack(df[col_a].values).astype(float)
    arr_b = np.stack(df[col_b].values).astype(float)

    # Compute absolute percentage difference
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress warnings
        abs_pct_diff = np.abs(arr_a - arr_b) / np.abs(arr_a) * 100
        abs_pct_diff[np.isnan(abs_pct_diff)] = 0  # handle division by zero

    # Assign back as a column of lists
    df[new_col_name] = list(abs_pct_diff)

    return df


# In[21]:


## Apply add_embedding_differences
## cls_raw_EMB
final_REPs_train = add_embedding_absolute_percentage_difference_vec(final_REPs_train, 'cls_raw_EMB', 'cls_raw_EMB_FT', new_col_name='cls_raw_EMB_diff')

## cls_t_EMB
final_REPs_train = add_embedding_absolute_percentage_difference_vec(final_REPs_train, 'cls_t_EMB', 'cls_t_EMB_FT', new_col_name='cls_t_EMB_diff')

## aa_EMB
final_REPs_train = add_embedding_absolute_percentage_difference_vec(final_REPs_train, 'aa_EMB', 'aa_EMB_FT', new_col_name='aa_EMB_diff')

#Inpect
print(final_REPs_train)


# In[25]:


## Funtion 'plot_mean_embedding_difference'
def plot_mean_embedding_difference(df, diff_column):
    """
    Compute and plot the mean vector of embedding differences across sequences,
    and save the plot as a PNG with the diff_column name in the filename.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the difference vectors column
    diff_column : str
        Name of the column containing embedding difference vectors
    figsize : tuple, optional
        Figure size for the plot
    
    Returns:
    --------
    mean_vector : np.ndarray
        The mean vector across all sequences
    """
    # Convert column of vectors into 2D array
    all_vectors = np.stack(df[diff_column].values)
    
    # Compute mean vector across sequences
    mean_vector = np.mean(all_vectors, axis=0)
    
    # Plot mean vector
    plt.figure(figsize=(10,3))
    plt.bar(range(len(mean_vector)), mean_vector, color='black', alpha=0.7)
    plt.xlabel("Embedding Dimension", fontsize=16)
    plt.ylabel("Mean Absolute Difference (%)", fontsize=16)
    plt.title(f"Mean Magnitude of Change Across Dimensions: '{diff_column}'", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Save plot in current directory
    filename = f"mean_embedding_difference_{diff_column}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()
    
    return mean_vector


# In[26]:


## Apply to CLS_raw_EMB
mean_vec = plot_mean_embedding_difference(final_REPs_train, 'cls_raw_EMB_diff')


# In[27]:


## Apply to CLS_t_EMB
mean_vec = plot_mean_embedding_difference(final_REPs_train, 'cls_t_EMB_diff')


# In[28]:


## Apply to AA_EMB
ean_vec = plot_mean_embedding_difference(final_REPs_train, 'aa_EMB_diff')

