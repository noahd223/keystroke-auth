import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib  # For saving the scaler
import os      # To handle file paths
import glob    # To find all files matching a pattern

# --- Configuration ---
# NEW: Point to the directory containing all user CSV files.
# IMPORTANT: Make sure your original 'noah_keystroke_data.csv' is also in this directory!
RAW_DATA_DIR = 'keystroke_data'  
PROCESSED_DATA_FILE = './processed_data/processed_data.pt' # Output file for PyTorch
SCALER_FILE = './processed_data/scaler.gz' # File to save the feature scaler
SEQUENCE_LENGTH = 40  # How many consecutive keystrokes to use for one training sample

# The features we will use to train the model
FEATURES = ['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time']

def preprocess_data():
    """
    Loads all raw keystroke data from a directory, combines it, cleans it,
    normalizes it, creates sequences, and saves it for PyTorch model training.
    """
    print("Starting data preprocessing...")

    # 1. Load and combine all raw data files from the directory
    try:
        # Create a pattern to find all .csv files in the directory
        path_pattern = os.path.join(RAW_DATA_DIR, '*.csv')
        all_csv_files = glob.glob(path_pattern)

        if not all_csv_files:
            print(f"Error: No CSV files were found in the directory '{RAW_DATA_DIR}'.")
            print("Please ensure your data files are in the correct location.")
            return

        # Read each CSV file and store its DataFrame in a list
        list_of_dfs = [pd.read_csv(f) for f in all_csv_files]
        
        # Combine all the DataFrames into one large DataFrame
        df = pd.concat(list_of_dfs, ignore_index=True)
        print(f"Loaded and combined {len(all_csv_files)} files with a total of {len(df)} keystroke events.")

    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return

    # 2. Data Cleaning
    # The first keystroke in any given prompt has flight times of 0, which isn't
    # meaningful data. We'll remove these rows.
    initial_rows = len(df)
    df = df[df['p2p_time'] > 0]
    # Also drop rows with extreme outliers, which can happen with long pauses.
    # We'll cap flight times at 3 seconds (3000 ms).
    df = df[df['r2p_time'] < 3000]
    print(f"Removed {initial_rows - len(df)} initial keystrokes and outliers.")
    
    if df.empty:
        print("No data left after cleaning. Exiting.")
        return

    # 3. Feature Scaling (Normalization)
    # We scale the features to be between 0 and 1. This helps the neural network
    # train more effectively.
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, SCALER_FILE)
    print(f"Features normalized and scaler saved to '{SCALER_FILE}'.")

    # 4. Create Sequences
    # We'll group the data by user and the prompt they typed, then create
    # overlapping sequences of keystrokes.
    sequences = []
    labels = []
    
    # Create a mapping from user ID (string) to a unique integer
    user_ids = df['user'].unique()
    user_to_id = {name: i for i, name in enumerate(user_ids)}
    print(f"Found {len(user_ids)} unique users. Mapping: {user_to_id}")

    # Group by user and prompt to create sequences from each typing session
    grouped = df.groupby(['user', 'prompt'])

    for _, group in grouped:
        feature_data = group[FEATURES].values
        user_name = group['user'].iloc[0]
        label = user_to_id[user_name]

        # Slide a window over the keystroke events to create sequences
        for i in range(len(feature_data) - SEQUENCE_LENGTH + 1):
            seq = feature_data[i : i + SEQUENCE_LENGTH]
            sequences.append(seq)
            labels.append(label)

    if not sequences:
        print("Could not create any sequences. Check if there's enough data for the given SEQUENCE_LENGTH.")
        return

    # 5. Convert to PyTorch Tensors
    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)
    
    print(f"Created {len(X)} sequences of length {SEQUENCE_LENGTH}.")
    print(f"Shape of feature tensor (X): {X.shape}")
    print(f"Shape of label tensor (y): {y.shape}")

    # 6. Save Processed Data
    data_to_save = {
        'sequences': X,
        'labels': y,
        'user_to_id': user_to_id,
        'features': FEATURES
    }
    torch.save(data_to_save, PROCESSED_DATA_FILE)
    print(f"Processed data saved to '{PROCESSED_DATA_FILE}'.")
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    # Before running, ensure the 'synthetic_data' directory exists and
    # contains all the CSV files you want to process (including your own).
    preprocess_data()
