import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import glob
import random

# --- Configuration ---
INPUT_DIR = 'keystroke_data/'
PROCESSED_DIR = 'processed_data/'
SEQUENCE_LENGTH = 100 
FEATURES_TO_USE = ['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time']

def create_sequences(df):
    """
    Converts the raw dataframe into a dictionary of sequences for each user and prompt.
    Returns a dict: {user: {prompt: [[feature1, feature2, ...], ...]}}
    """
    sequences = {}
    print(f"Extracting features: {FEATURES_TO_USE}")
    for (user, prompt), group in df.groupby(['user', 'prompt']):
        if user not in sequences:
            sequences[user] = {}
        features = group[FEATURES_TO_USE].values
        sequences[user][prompt] = features
    return sequences

def create_pairs(sequences):
    """
    Generates a BALANCED set of positive (same user) and negative (different user) pairs.
    """
    positive_pairs = []
    negative_pairs = []
    users = list(sequences.keys())

    print("Generating positive pairs...")
    for user in tqdm(users, desc="Processing users for positive pairs"):
        user_prompts = list(sequences[user].keys())
        if len(user_prompts) < 2:
            continue
        for prompt1, prompt2 in itertools.combinations(user_prompts, 2):
            positive_pairs.append([sequences[user][prompt1], sequences[user][prompt2]])

    num_positive_pairs = len(positive_pairs)
    print(f"Generated {num_positive_pairs} positive pairs.")
    print("Generating an equal number of negative pairs...")
    
    # --- EDITED: Generate an equal number of negative pairs ---
    while len(negative_pairs) < num_positive_pairs:
        # Pick two different random users
        u1, u2 = random.sample(users, 2)
        
        # Pick a random prompt from each of the selected users
        prompt1 = random.choice(list(sequences[u1].keys()))
        prompt2 = random.choice(list(sequences[u2].keys()))
        
        negative_pairs.append([sequences[u1][prompt1], sequences[u2][prompt2]])

    return positive_pairs, negative_pairs


def normalize_and_pad(pairs, sequence_length):
    """
    Normalizes and pads/truncates all sequences in the pairs to a fixed length.
    """
    processed_pairs = []
    scaler = StandardScaler()

    all_sequences = [seq for pair in pairs for seq in pair]
    if not all_sequences: return np.array([])
    
    flat_list = [item for sublist in all_sequences for item in sublist]
    if not flat_list: return np.array([])

    scaler.fit(np.array(flat_list))
    
    print("Normalizing and padding sequences...")
    for pair in tqdm(pairs, desc="Processing pairs"):
        padded_pair = []
        for seq in pair:
            normalized_seq = scaler.transform(seq)
            if len(normalized_seq) < sequence_length:
                pad_width = sequence_length - len(normalized_seq)
                padded_seq = np.pad(normalized_seq, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
            else:
                padded_seq = normalized_seq[:sequence_length]
            padded_pair.append(padded_seq)
        processed_pairs.append(padded_pair)
        
    return np.array(processed_pairs)


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    if not csv_files:
        print(f"Error: No .csv files found in the directory '{INPUT_DIR}'.")
        return

    print(f"Found {len(csv_files)} CSV files to process in '{INPUT_DIR}'.")
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    print(f"Total rows loaded: {len(df)}")

    if df['user'].nunique() < 2:
        print("Error: Need data from at least two different users to create negative pairs.")
        return

    sequences = create_sequences(df)
    positive_pairs, negative_pairs = create_pairs(sequences)
    
    print(f"\nGenerated {len(positive_pairs)} positive pairs.")
    print(f"Generated {len(negative_pairs)} negative pairs. (Balanced)")

    X_pos = normalize_and_pad(positive_pairs, SEQUENCE_LENGTH)
    X_neg = normalize_and_pad(negative_pairs, SEQUENCE_LENGTH)
    
    y_pos = np.ones(len(X_pos))
    y_neg = np.zeros(len(X_neg))
    
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    print(f"\nSaving processed data to '{PROCESSED_DIR}'...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    np.save(os.path.join(PROCESSED_DIR, 'X_processed.npy'), X)
    np.save(os.path.join(PROCESSED_DIR, 'y_processed.npy'), y)
    
    print("\nPreprocessing complete!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == '__main__':
    main()
