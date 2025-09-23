#### Imports
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

#Data load
file_path= r'/home/afarodrigues/datasets/Filipa/41587_2020_793_MOESM3_ESM.csv'
df = pd.read_csv(file_path)
print(df.head())

#### Overall statistics before duplicates eliminatio

# Count number of sequences in each partition
unique_values_in_partition = df['partition'].value_counts().reset_index()

# Rename the columns
unique_values_in_partition.columns = ['partition', 'num_sequences']

# Count the minimum number of mutations in each partition
min_mutations_per_partition = df.groupby('partition')['num_mutations'].min().reset_index()

# Rename the columns
min_mutations_per_partition.columns = ['partition', 'min_num_mutations']

# Merge the two dfs on the 'partition' column
result_df = pd.merge(unique_values_in_partition, min_mutations_per_partition, on='partition')

# Count the maximum number of mutations in each partition
max_mutations_per_partition = df.groupby('partition')['num_mutations'].max().reset_index()
# Rename the columns
max_mutations_per_partition.columns = ['partition', 'max_num_mutations']
# Merge the new information with the existing df on the 'partition' column
result_df = pd.merge(result_df, max_mutations_per_partition, on='partition')

# Count the viable and non-viable sequences in each partition
# Count occurrences of 'TRUE' in 'is_viable' column for each partition
true_counts_per_partition = df[df['is_viable'] == True].groupby('partition').size().reset_index(name='Viable')

# Merge the new information with the existing df on the 'partition' column
result_df = pd.merge(result_df, true_counts_per_partition, on='partition', how='left')

# Count occurrences of 'FALSE' in 'is_viable' column for each partition
false_counts_per_partition = df[df['is_viable'] == False].groupby('partition').size().reset_index(name='Non-viable')
# Merge the new information with the existing df on the 'partition' column
result_df = pd.merge(result_df, false_counts_per_partition, on='partition', how='left')

# Replace NaNs with 0 in the 'Viable' and 'Non-viable' columns
result_df['Viable'] = result_df['Viable'].fillna(0)
result_df['Non-viable'] = result_df['Non-viable'].fillna(0)

# Final df
summary_df = result_df

#Export
file_path = r'/home/afarodrigues/embCOMP/outputs_csv/data_summary_before_dup_elimination.csv'
summary_df.to_csv(file_path)
print('Saved AAV_data_summary_before_dup_elimination.csv')

# ### Handling duplicates

# Find all occurrences of duplicates (including first occurrence)
duplicates_all = df[df['sequence'].duplicated(keep=False)]

print('duplicates:')
print(duplicates_all)
print('')

#Export
file_path = r'/home/afarodrigues/embCOMP/outputs_csv/duplicates_all.csv'
duplicates_all.to_csv(file_path, index=True)
print('Saved duplicates_all.csv')

# Sort to pair
duplicates_all_sorted = duplicates_all.sort_values(by='sequence', ascending=True)

# Ensure even number of rows
if len(duplicates_all_sorted) % 2 != 0:
    raise ValueError("DataFrame should have an even number of rows for pairing.")

# Extract all pairs
pairs = [tuple(sorted(duplicates_all_sorted['partition'].iloc[i:i+2])) for i in range(0, len(duplicates_all_sorted), 2)]

# Count occurrences
pair_counts = Counter(pairs)

# Convert to DataFrame
pair_counts = pd.DataFrame(pair_counts.items(), columns=['Pair', 'Count'])

print('pair counts:')
print(pair_counts)
print('')

# Removal of some partitions 'stop', 'wild-type', 'previous_chip_viables' and 'previous_chip_nonviables'
df = df[~df['partition'].isin(['stop', 'wild_type', 'previous_chip_nonviable', 'previous_chip_viable'])]

# Find all occurrences of duplicates (including first occurrence)
duplicates_all = df[df['sequence'].duplicated(keep=False)]

# Sort to pair
duplicates_all_sorted = duplicates_all.sort_values(by='sequence', ascending=True)

# Ensure even number of rows
if len(duplicates_all_sorted) % 2 != 0:
    raise ValueError("DataFrame should have an even number of rows for pairing.")

# Extract all pairs
pairs = [tuple(sorted(duplicates_all_sorted['partition'].iloc[i:i+2])) for i in range(0, len(duplicates_all_sorted), 2)]

# Count occurrences
pair_counts = Counter(pairs)

# Convert to DataFrame
pair_counts = pd.DataFrame(pair_counts.items(), columns=['Pair', 'Count'])

print('pair counts:')
print(pair_counts)
print('')

# Removal partition 'singles'
df = df[~df['partition'].isin(['singles'])]

# Remove rows where 'partition' is 'random_doubles' in duplicates_all_sorted
# Create a df with the rows to remove
random_doubles_df = duplicates_all_sorted[duplicates_all_sorted['partition'] == 'random_doubles'].copy()
random_doubles_df

# Remove the rows in 'df' that match the index of 'random_doubles_df'
df = df[~df.index.isin(random_doubles_df.index)].copy()

# Last search for duplicates (should be 0)
# Find all occurrences of duplicates (including first occurrence)
duplicates_all = df[df['sequence'].duplicated(keep=False)]

print('duplicates:')
print(duplicates_all)
print('')

# Count number of sequences in each partition
unique_values_in_partition = df['partition'].value_counts().reset_index()

# Rename the columns
unique_values_in_partition.columns = ['partition', 'num_sequences']

# Count the minimum number of mutations in each partition
min_mutations_per_partition = df.groupby('partition')['num_mutations'].min().reset_index()

# Rename the columns
min_mutations_per_partition.columns = ['partition', 'min_num_mutations']

# Merge the two dfs on the 'partition' column
result_df = pd.merge(unique_values_in_partition, min_mutations_per_partition, on='partition')

# Count the maximum number of mutations in each partition
max_mutations_per_partition = df.groupby('partition')['num_mutations'].max().reset_index()

# Rename the columns
max_mutations_per_partition.columns = ['partition', 'max_num_mutations']

# Merge the new information with the existing df on the 'partition' column
result_df = pd.merge(result_df, max_mutations_per_partition, on='partition')

# Count the viable and non-viable sequences in each partition
# Count occurrences of 'TRUE' in 'is_viable' column for each partition
true_counts_per_partition = df[df['is_viable'] == True].groupby('partition').size().reset_index(name='Viable')

# Merge the new information with the existing df on the 'partition' column
result_df = pd.merge(result_df, true_counts_per_partition, on='partition', how='left')

# Count occurrences of 'FALSE' in 'is_viable' column for each partition
false_counts_per_partition = df[df['is_viable'] == False].groupby('partition').size().reset_index(name='Non-viable')

# Merge the new information with the existing df on the 'partition' column
result_df = pd.merge(result_df, false_counts_per_partition, on='partition', how='left')

# Replace NaNs with 0 in the 'Viable' and 'Non-viable' columns
result_df['Viable'] = result_df['Viable'].fillna(0)
result_df['Non-viable'] = result_df['Non-viable'].fillna(0)

# Final df
file_path = r'/home/afarodrigues/embCOMP/outputs_csv/data_summary_after_dup_elimination.csv'
result_df.to_csv(file_path)
print('Saved AAV_data_summary_after_dup_elimination.csv')

# Export to csv to save (overwrite aav_data.csv, now without duplicates)
file_path = r'/home/afarodrigues/embCOMP/outputs_csv/data.csv'
df.to_csv(file_path, index=False)
print('Saved aav_data.csv (overwrite)')

##### Reconstruct sequences

## Function to get specific data partitions (either full sequences or just the fragment)

## Sequences for pre and post fragment
pre_sequence = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMIT"
post_sequence = "QAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

def get_partition_data(file_name, partitions_to_get, get_full_sequences=False):
    # Read the csv containing the data
    df = pd.read_csv(file_name)

    print("Should do full:", get_full_sequences)

    all_data = []

    # For row in the dataset
    for index, row in df.iterrows():

        p = row["partition"]

        # If the partition is one of those we want
        if p in partitions_to_get:
            s = row["sequence"]
            p = row['partition']
            l = row["is_viable"]

            # Turn TRUE and FALSE into 1s and 0s
            try:
                l = int(l)
            except:
                print("ERROR IN COLUMN")

            if get_full_sequences:
                s = pre_sequence + s + post_sequence

            # Append the data to the list
            all_data.append({'sequence': s, 'partition': p, 'is_viable': l})

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(all_data)

    return result_df


# Use of 'get_partition_data' to get all (full) sequences from all partitions
partitions = ['designed', 'rand', 'random_doubles','single', 'singles', 'stop','previous_chip_viable', 
              'previous_chip_nonviable', 'lr_designed_plus_rand_train_walked', 'lr_rand_doubles_plus_single_walked',
              'rnn_standard_walked', 'rnn_standard_walked', 'cnn_rand_doubles_plus_single_walked', 
              'cnn_standard_walked', 'rnn_designed_plus_rand_train_walked', 'rnn_rand_doubles_plus_singles_seed',
              'cnn_designed_plus_rand_train_seed','rnn_designed_plus_rand_train_seed', 'lr_standard_seed', 
              'rnn_rand_doubles_plus_singles_seed','cnn_rand_doubles_plus_single_seed', 'cnn_standard_seed', 
              'cnn_designed_plus_rand_train_seed', 'lr_designed_plus_rand_train_seed', 'rnn_standard_seed', 
              'lr_rand_doubles_plus_single_seed', 'lr_standard_walked', 'rnn_rand_doubles_plus_singles_walked',
             'cnn_designed_plus_rand_train_walked']

# Get all the (seq, label) pairs present in the designed, rand, and single partitions, with the complete sequences.
file_path = r'/home/afarodrigues/embCOMP/outputs_csv/data.csv'
sequences_partition_labels = get_partition_data(file_path, partitions, get_full_sequences=True)

# Function to generate sequence IDs based on partition name and counter
def generate_sequence_id(row):
    partition = row['partition']
    if partition not in partition_counters:
        partition_counters[partition] = 1
    else:
        partition_counters[partition] += 1
    return f"{partition}_{partition_counters[partition]}"


## use 'generate_sequence_id' function to add sequence ID labels

# Initialize a dictionary to store the counter for each partition
partition_counters = {}

# Add the 'sequence_ID' column using the function
sequences_partition_labels.insert(0, 'sequence_ID', sequences_partition_labels.apply(generate_sequence_id, axis=1))

# Rename columns
sequences_partition_labels.rename(columns={'sequence_ID':'Sequence_ID','sequence': 'Sequence', 'partition':'Subset','is_viable': 'Viability'}, inplace=True)

# Convert all sequences in the 'Sequence' column to uppercase
sequences_partition_labels['Sequence'] = sequences_partition_labels['Sequence'].str.upper()

# Rename columns
sequences_partition_labels.rename(columns={'sequence_ID':'Sequence_ID','sequence': 'Sequence', 'partition':'Subset','is_viable': 'Viability'}, inplace=True)

# Convert all sequences in the 'Sequence' column to uppercase
sequences_partition_labels['Sequence'] = sequences_partition_labels['Sequence'].str.upper()

# Export
file_path = r'/home/afarodrigues/embCOMP/outputs_csv/data_2.csv'
sequences_partition_labels.to_csv(file_path, index=True)

print('')
print('Saved data_2.csv')
print('')


##### Data split (70-10-20)
file_path = r'/home/afarodrigues/embCOMP/outputs_csv/data_2.csv'
df = pd.read_csv(file_path)

# Split with stratification

# Define the label column to stratify by
label_col = 'Viability' 

# First, split off the test set (20%)
df_train_val, df_test = train_test_split(
    df, test_size=0.20, stratify=df[label_col], random_state=42
)
# Now split the remaining 90% into train (70%) and val (10%) → (70/80 ≈ 0.875)
df_train, df_val = train_test_split(
    df_train_val, test_size=0.125, stratify=df_train_val[label_col], random_state=42
)
# Check sizes
print(f"Train: {len(df_train)} rows")
print(f"Validation: {len(df_val)} rows")
print(f"Test: {len(df_test)} rows")

# Save to CSV
df_train.to_csv(r'/home/afarodrigues/datasets/Filipa/data_train.csv', index=False)
df_val.to_csv(r'/home/afarodrigues/datasets/Filipa/data_val.csv', index=False)
df_test.to_csv(r'/home/afarodrigues/datasets/Filipa/data_test.csv', index=False)

print('')
print('Finished data split')
