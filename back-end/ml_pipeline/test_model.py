#!/usr/bin/env python3
"""
Fixed test script that uses the same preprocessing as training.
"""

import torch
import numpy as np
import pandas as pd
from train_model import ImprovedSiameseNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
import os
import random
from sklearn.model_selection import train_test_split

def load_test_data():
    """Load the processed test data (same as original test_model.py)."""
    try:
        # Load processed data
        X = np.load('../processed_data/X_processed.npy')
        y = np.load('../processed_data/y_processed.npy')
        
        # Split into train/test (same as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Loaded test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None

def load_model():
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture (must match training)
    input_dim = 4
    hidden_dim = 64
    embedding_dim = 32
    
    model = ImprovedSiameseNetwork(input_dim, hidden_dim, embedding_dim)
    model.load_state_dict(torch.load('../saved_models/improved_siamese_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model, device

def test_on_original_processed_data():
    """Test on the original processed data to verify model works correctly."""
    print("="*60)
    print("TESTING ON ORIGINAL PROCESSED DATA")
    print("="*60)
    
    # Load model and data
    model, device = load_model()
    X_test, y_test = load_test_data()
    
    if X_test is None or y_test is None:
        print("Could not load test data. Exiting.")
        return
    
    # Convert to tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Get predictions
    print("Making predictions on test data...")
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test_tensor[i:i+batch_size]
            # Split paired sequences
            seq1 = batch_x[:, 0]  # First sequence of each pair
            seq2 = batch_x[:, 1]  # Second sequence of each pair
            
            batch_pred = model(seq1, seq2)
            probabilities.extend(batch_pred.cpu().numpy())
    
    probabilities = np.array(probabilities)
    predictions = (probabilities > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Print results
    print(f"\n📊 ORIGINAL DATA PERFORMANCE:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    
    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives:  {cm[1,1]}")
    
    # Calculate error rates
    far = cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
    frr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    print(f"\n🚨 ERROR RATES:")
    print(f"   False Accept Rate (FAR):  {far:.4f} ({far*100:.2f}%)")
    print(f"   False Reject Rate (FRR):  {frr:.4f} ({frr*100:.2f}%)")
    
    # Probability distribution analysis
    same_user_probs = probabilities[y_test == 1]
    diff_user_probs = probabilities[y_test == 0]
    
    print(f"\n📈 PROBABILITY DISTRIBUTIONS:")
    print(f"   Same User Pairs: μ={same_user_probs.mean():.4f}, σ={same_user_probs.std():.4f}")
    print(f"   Different User Pairs: μ={diff_user_probs.mean():.4f}, σ={diff_user_probs.std():.4f}")
    
    return accuracy > 0.8  # Return True if model performs well

def load_and_preprocess_synthetic_data():
    """Load synthetic data and preprocess it using the same method as training."""
    print("\n" + "="*60)
    print("LOADING AND PREPROCESSING SYNTHETIC DATA")
    print("="*60)
    
    data_dir = '../keystroke_data'
    features = ['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time']
    
    # Load all synthetic user data
    user_data = {}
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'user_' in f]
    
    print(f"Loading {len(csv_files)} synthetic user files...")
    
    for filename in csv_files:
        user_id = filename.replace('_keystrokes.csv', '')
        filepath = os.path.join(data_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            # Remove outliers (same as training preprocessing)
            df = df[df[features].max(axis=1) < 5000]
            df = df[df[features].min(axis=1) > 0]
            
            if len(df) >= 50:  # Only keep users with sufficient data
                user_data[user_id] = df[features].values
                print(f"   ✓ {user_id}: {len(df)} keystrokes")
            else:
                print(f"   ❌ {user_id}: insufficient data ({len(df)} keystrokes)")
        except Exception as e:
            print(f"   ❌ Failed to load {filename}: {e}")
    
    if len(user_data) < 2:
        print("Not enough users with sufficient data!")
        return None, None
    
    # Create test pairs
    print(f"\nCreating test pairs from {len(user_data)} users...")
    test_pairs = []
    labels = []
    users = list(user_data.keys())
    
    pairs_per_user = 20  # Reduce to avoid memory issues
    
    for user in users:
        user_sequences = user_data[user]
        
        # Create same-user pairs
        for _ in range(pairs_per_user):
            if len(user_sequences) < 100:  # Need at least 100 keystrokes for 2 sequences
                continue
                
            # Split randomly
            indices = np.random.permutation(len(user_sequences))
            mid = len(indices) // 2
            
            seq1 = user_sequences[indices[:mid]]
            seq2 = user_sequences[indices[mid:]]
            
            test_pairs.append([seq1, seq2])
            labels.append(1)  # Same user
        
        # Create different-user pairs
        other_users = [u for u in users if u != user]
        for _ in range(pairs_per_user):
            if not other_users:
                continue
                
            other_user = np.random.choice(other_users)
            
            # Get random subsequences
            seq1_len = min(50, len(user_sequences))
            seq2_len = min(50, len(user_data[other_user]))
            
            seq1_start = np.random.randint(0, len(user_sequences) - seq1_len + 1)
            seq2_start = np.random.randint(0, len(user_data[other_user]) - seq2_len + 1)
            
            seq1 = user_sequences[seq1_start:seq1_start + seq1_len]
            seq2 = user_data[other_user][seq2_start:seq2_start + seq2_len]
            
            test_pairs.append([seq1, seq2])
            labels.append(0)  # Different users
    
    print(f"Created {len(test_pairs)} test pairs ({sum(labels)} same-user, {len(labels) - sum(labels)} different-user)")
    
    if not test_pairs:
        return None, None
    
    # Normalize and pad sequences (same as training)
    sequence_length = 50
    processed_pairs = []
    
    # Collect all sequences for normalization
    all_sequences = []
    for pair in test_pairs:
        for seq in pair:
            # Pad or truncate
            if len(seq) < sequence_length:
                padded = np.pad(seq, ((0, sequence_length - len(seq)), (0, 0)), 'constant')
            else:
                padded = seq[:sequence_length]
            all_sequences.append(padded)
    
    all_sequences = np.array(all_sequences)
    
    # Normalize each feature separately using RobustScaler (same as training)
    print("Normalizing features...")
    for feature_idx in range(4):  # 4 features
        scaler = RobustScaler()
        feature_values = all_sequences[:, :, feature_idx].flatten()
        non_zero_mask = feature_values != 0
        
        if np.sum(non_zero_mask) > 0:
            scaler.fit(feature_values[non_zero_mask].reshape(-1, 1))
            transformed = scaler.transform(feature_values.reshape(-1, 1)).flatten()
            all_sequences[:, :, feature_idx] = transformed.reshape(len(all_sequences), sequence_length)
    
    # Reconstruct pairs
    for i in range(0, len(all_sequences), 2):
        processed_pairs.append([all_sequences[i], all_sequences[i+1]])
    
    return np.array(processed_pairs), np.array(labels)

def test_on_synthetic_data():
    """Test the model on properly preprocessed synthetic data."""
    print("\n" + "="*60)
    print("TESTING ON SYNTHETIC DATA (PROPER PREPROCESSING)")
    print("="*60)
    
    # Load and preprocess synthetic data
    X_synthetic, y_synthetic = load_and_preprocess_synthetic_data()
    
    if X_synthetic is None:
        print("Could not create test data from synthetic users!")
        return
    
    # Load model
    model, device = load_model()
    
    # Convert to tensors and test
    X_tensor = torch.tensor(X_synthetic, dtype=torch.float32).to(device)
    
    print(f"Testing on {len(X_synthetic)} synthetic pairs...")
    probabilities = []
    
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[i:i+batch_size]
            seq1 = batch_x[:, 0]  # First sequence of each pair
            seq2 = batch_x[:, 1]  # Second sequence of each pair
            
            batch_pred = model(seq1, seq2)
            probabilities.extend(batch_pred.cpu().numpy())
    
    probabilities = np.array(probabilities)
    predictions = (probabilities > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_synthetic, predictions)
    if len(np.unique(predictions)) > 1:
        precision = precision_score(y_synthetic, predictions)
        recall = recall_score(y_synthetic, predictions)
        f1 = f1_score(y_synthetic, predictions)
        auc = roc_auc_score(y_synthetic, probabilities)
    else:
        precision = recall = f1 = auc = 0.0
    
    cm = confusion_matrix(y_synthetic, predictions)
    
    # Print results
    print(f"\n📊 SYNTHETIC DATA PERFORMANCE:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    
    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives:  {cm[1,1]}")
    
    # Error rates
    far = cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
    frr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    print(f"\n🚨 ERROR RATES:")
    print(f"   False Accept Rate (FAR):  {far:.4f} ({far*100:.2f}%)")
    print(f"   False Reject Rate (FRR):  {frr:.4f} ({frr*100:.2f}%)")
    
    # Probability distributions
    same_user_probs = probabilities[y_synthetic == 1]
    diff_user_probs = probabilities[y_synthetic == 0]
    
    print(f"\n📈 PROBABILITY DISTRIBUTIONS:")
    print(f"   Same User Pairs: μ={same_user_probs.mean():.4f}, σ={same_user_probs.std():.4f}")
    print(f"   Different User Pairs: μ={diff_user_probs.mean():.4f}, σ={diff_user_probs.std():.4f}")

if __name__ == "__main__":
    print("🔬 COMPREHENSIVE MODEL TESTING")
    print("="*60)
    
    # First, test on original data to verify model works
    print("Step 1: Verifying model works on original test data...")
    model_works = test_on_original_processed_data()
    
    if model_works:
        print("\n✅ Model works correctly on original data!")
        print("Step 2: Testing on synthetic data with proper preprocessing...")
        test_on_synthetic_data()
    else:
        print("\n❌ Model has issues with original data. Check model training.") 