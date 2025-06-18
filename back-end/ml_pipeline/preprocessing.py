import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import os
import glob
import random

# --- Configuration ---
INPUT_DIR = '../keystroke_data/'
PROCESSED_DIR = '../processed_data/'
SEQUENCE_LENGTH = 50  # Reduced from 100 to focus on shorter, more meaningful sequences
FEATURES_TO_USE = ['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time']

def create_sequences(df):
    """
    Converts the raw dataframe into sequences for each user and prompt.
    Improved to handle outliers and validate data quality.
    """
    sequences = {}
    print(f"Extracting features: {FEATURES_TO_USE}")
    
    # Remove obvious outliers (timing values > 5 seconds are likely errors)
    df = df[df[FEATURES_TO_USE].max(axis=1) < 5000]
    df = df[df[FEATURES_TO_USE].min(axis=1) > 0]
    
    print(f"After outlier removal: {len(df)} rows")
    
    for (user, prompt), group in df.groupby(['user', 'prompt']):
        if len(group) < 10:  # Skip very short sequences
            continue
            
        if user not in sequences:
            sequences[user] = {}
        
        features = group[FEATURES_TO_USE].values
        sequences[user][prompt] = features
    
    return sequences

def create_balanced_pairs(sequences, max_pairs_per_user=100):
    """
    Creates a more balanced dataset with controlled sampling.
    """
    positive_pairs = []
    negative_pairs = []
    users = list(sequences.keys())
    
    print(f"Creating pairs from {len(users)} users")
    
    # Generate positive pairs (same user, different prompts)
    for user in tqdm(users, desc="Generating positive pairs"):
        user_prompts = list(sequences[user].keys())
        if len(user_prompts) < 2:
            continue
            
        # Limit pairs per user to prevent dominance
        user_positive_pairs = list(itertools.combinations(user_prompts, 2))
        if len(user_positive_pairs) > max_pairs_per_user:
            user_positive_pairs = random.sample(user_positive_pairs, max_pairs_per_user)
        
        for prompt1, prompt2 in user_positive_pairs:
            positive_pairs.append([sequences[user][prompt1], sequences[user][prompt2]])
    
    print(f"Generated {len(positive_pairs)} positive pairs")
    
    # Generate negative pairs (different users)
    target_negative_pairs = len(positive_pairs)
    print(f"Generating {target_negative_pairs} negative pairs...")
    
    while len(negative_pairs) < target_negative_pairs:
        u1, u2 = random.sample(users, 2)
        prompt1 = random.choice(list(sequences[u1].keys()))
        prompt2 = random.choice(list(sequences[u2].keys()))
        negative_pairs.append([sequences[u1][prompt1], sequences[u2][prompt2]])
    
    return positive_pairs, negative_pairs

def improved_normalize_and_pad(pairs, sequence_length):
    """
    Improved normalization that handles features separately and uses robust scaling.
    """
    if not pairs:
        return np.array([])
    
    processed_pairs = []
    
    # Collect all sequences and flatten for normalization
    all_sequences = [seq for pair in pairs for seq in pair]
    if not all_sequences:
        return np.array([])
    
    # Pad/truncate sequences first
    padded_sequences = []
    for seq in all_sequences:
        if len(seq) < sequence_length:
            # Pad with zeros
            pad_width = sequence_length - len(seq)
            padded_seq = np.pad(seq, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
        else:
            # Truncate to sequence_length
            padded_seq = seq[:sequence_length]
        padded_sequences.append(padded_seq)
    
    # Convert to numpy array for easier manipulation
    all_padded = np.array(padded_sequences)  # Shape: (n_sequences, sequence_length, n_features)
    
    # Normalize each feature separately using RobustScaler (less sensitive to outliers)
    scalers = []
    for feature_idx in range(len(FEATURES_TO_USE)):
        scaler = RobustScaler()
        # Extract all values for this feature across all sequences and time steps
        feature_values = all_padded[:, :, feature_idx].flatten()
        # Only fit on non-zero values (ignore padding)
        non_zero_mask = feature_values != 0
        if np.sum(non_zero_mask) > 0:
            scaler.fit(feature_values[non_zero_mask].reshape(-1, 1))
            # Transform all values (including zeros)
            transformed = scaler.transform(feature_values.reshape(-1, 1)).flatten()
            all_padded[:, :, feature_idx] = transformed.reshape(all_padded.shape[0], sequence_length)
        scalers.append(scaler)
    
    print("Normalization complete. Creating pairs...")
    
    # Reconstruct pairs
    seq_idx = 0
    for pair in tqdm(pairs, desc="Reconstructing pairs"):
        pair_sequences = []
        for _ in range(2):  # Each pair has 2 sequences
            pair_sequences.append(all_padded[seq_idx])
            seq_idx += 1
        processed_pairs.append(pair_sequences)
    
    return np.array(processed_pairs)

def validate_processed_data(X, y):
    """
    Validate the processed data for common issues.
    """
    print("\nValidating processed data...")
    
    # Check for NaN or infinite values
    if np.isnan(X).any():
        print("WARNING: NaN values found in X")
    if np.isinf(X).any():
        print("WARNING: Infinite values found in X")
    
    # Check data distribution
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X min: {X.min():.4f}, max: {X.max():.4f}")
    print(f"X mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"y distribution: {np.bincount(y.astype(int))}")
    
    # Check if features have reasonable variance
    feature_vars = np.var(X.reshape(-1, X.shape[-1]), axis=0)
    print(f"Feature variances: {feature_vars}")
    
    if np.any(feature_vars < 0.01):
        print("WARNING: Some features have very low variance")
    
    return True

def main():
    """
    Main function with improved preprocessing pipeline.
    """
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    if not csv_files:
        print(f"Error: No .csv files found in '{INPUT_DIR}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Load and combine all data
    df_list = []
    for file in csv_files:
        try:
            df_temp = pd.read_csv(file)
            df_list.append(df_temp)
        except Exception as e:
            print(f"Warning: Could not read {file}: {e}")
    
    if not df_list:
        print("Error: No valid CSV files found")
        return
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    # Check minimum number of users
    unique_users = df['user'].nunique()
    print(f"Number of unique users: {unique_users}")
    
    if unique_users < 3:
        print("Error: Need data from at least 3 different users for meaningful training")
        return
    
    # Create sequences
    sequences = create_sequences(df)
    
    # Create balanced pairs
    positive_pairs, negative_pairs = create_balanced_pairs(sequences)
    
    print(f"\nGenerated {len(positive_pairs)} positive pairs")
    print(f"Generated {len(negative_pairs)} negative pairs")
    
    # Process data
    X_pos = improved_normalize_and_pad(positive_pairs, SEQUENCE_LENGTH)
    X_neg = improved_normalize_and_pad(negative_pairs, SEQUENCE_LENGTH)
    
    if X_pos.size == 0 or X_neg.size == 0:
        print("Error: No valid pairs generated")
        return
    
    # Create labels
    y_pos = np.ones(len(X_pos))
    y_neg = np.zeros(len(X_neg))
    
    # Combine and shuffle
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Validate data
    validate_processed_data(X, y)
    
    # Save processed data
    print(f"\nSaving processed data to '{PROCESSED_DIR}'...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    np.save(os.path.join(PROCESSED_DIR, 'X_processed.npy'), X)
    np.save(os.path.join(PROCESSED_DIR, 'y_processed.npy'), y)
    
    print("\nImproved preprocessing complete!")
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")

if __name__ == '__main__':
    main() 