#!/usr/bin/env python3
"""
Comprehensive Model Testing Script
Tests the keystroke authentication model on both existing processed data 
and newly generated synthetic data with proper preprocessing.
"""

import torch
import numpy as np
import pandas as pd
from train_model import ImprovedSiameseNetwork
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import glob
import random

class ModelTester:
    """
    Comprehensive model testing class for keystroke authentication.
    """
    
    def __init__(self, model_path='../saved_models/improved_siamese_model.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 50
        self.features = ['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time']
        
    def load_model(self):
        """Load the trained model."""
        try:
            input_dim = 4
            hidden_dim = 64
            embedding_dim = 32
            
            self.model = ImprovedSiameseNetwork(input_dim, hidden_dim, embedding_dim)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_on_processed_data(self):
        """Test model on the existing processed data."""
        print("\n" + "="*60)
        print("🧪 TESTING ON EXISTING PROCESSED DATA")
        print("="*60)
        
        try:
            # Load processed data
            X = np.load('../processed_data/X_processed.npy')
            y = np.load('../processed_data/y_processed.npy')
            
            # Split into train/test (same as training)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Test data shape: {X_test.shape}")
            print(f"📊 Test labels shape: {y_test.shape}")
            print(f"📊 Positive samples: {sum(y_test)}, Negative samples: {len(y_test) - sum(y_test)}")
            
        except FileNotFoundError:
            print("❌ Processed data not found. Please run preprocessing first.")
            return None
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return None
        
        # Convert to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        # Get predictions
        print("🔮 Making predictions...")
        probabilities = []
        
        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(X_test), batch_size):
                batch_x = X_test_tensor[i:i+batch_size]
                seq1 = batch_x[:, 0]  # First sequence of each pair
                seq2 = batch_x[:, 1]  # Second sequence of each pair
                
                batch_pred = self.model(seq1, seq2)
                probabilities.extend(batch_pred.cpu().numpy())
        
        probabilities = np.array(probabilities)
        predictions = (probabilities > 0.5).astype(int)
        
        # Calculate and display metrics
        self._display_metrics(y_test, predictions, probabilities, "PROCESSED DATA")
        
        return probabilities.mean() > 0.1 and probabilities.std() > 0.1  # Check if model is working
    
    def load_synthetic_data(self):
        """Load and preprocess synthetic keystroke data."""
        print("\n" + "="*60)
        print("📁 LOADING SYNTHETIC DATA")
        print("="*60)
        
        data_dir = '../keystroke_data'
        
        # Get synthetic user files (both user_* and *_copy_* patterns)
        csv_files = (glob.glob(os.path.join(data_dir, 'user_*.csv')) + 
                    glob.glob(os.path.join(data_dir, '*_copy_*.csv')))
        
        if not csv_files:
            print("❌ No synthetic user data found!")
            return None
        
        print(f"📂 Found {len(csv_files)} synthetic user files")
        
        # Load data
        user_data = {}
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            # Extract user ID from different naming patterns
            if '_copy_' in filename:
                user_id = filename.split('_copy_')[0]  # Extract base username
            else:
                user_id = filename.replace('_keystrokes.csv', '')
            
            try:
                df = pd.read_csv(filepath)
                
                # Check if required columns exist
                if not all(col in df.columns for col in self.features):
                    print(f"⚠️ {user_id}: Missing required columns")
                    continue
                
                # Extract and clean features
                features_data = df[self.features].values
                
                # Remove outliers (same as training preprocessing)
                mask = (features_data > 0).all(axis=1) & (features_data < 5000).all(axis=1)
                features_data = features_data[mask]
                
                if len(features_data) >= 50:  # Minimum data requirement
                    # Accumulate data for users with multiple files
                    if user_id in user_data:
                        user_data[user_id] = np.vstack([user_data[user_id], features_data])
                        print(f"   ✅ {user_id} (+): {len(features_data)} more keystrokes (total: {len(user_data[user_id])})")
                    else:
                        user_data[user_id] = features_data
                        print(f"   ✅ {user_id}: {len(features_data)} valid keystrokes")
                else:
                    print(f"   ⚠️ {user_id}: Insufficient data ({len(features_data)} keystrokes)")
                    
            except Exception as e:
                print(f"   ❌ {user_id}: Error loading - {e}")
        
        if len(user_data) < 3:
            print("❌ Need at least 3 users with sufficient data!")
            return None
        
        print(f"✅ Successfully loaded {len(user_data)} users")
        return user_data
    
    def create_test_pairs_from_synthetic(self, user_data, pairs_per_user=30):
        """Create test pairs from synthetic data with proper preprocessing."""
        print(f"\n🔄 Creating test pairs ({pairs_per_user} per user)...")
        
        test_pairs = []
        labels = []
        users = list(user_data.keys())
        
        print(f"👥 Working with {len(users)} users...")
        
        for user in users:
            user_sequences = user_data[user]
            
            if len(user_sequences) < 100:  # Need enough data for splitting
                print(f"⚠️ Skipping {user}: insufficient data ({len(user_sequences)} keystrokes)")
                continue
            
            # Create same-user pairs
            for _ in range(pairs_per_user):
                # Randomly split user's data
                indices = np.random.permutation(len(user_sequences))
                split_point = len(indices) // 2
                
                seq1 = user_sequences[indices[:split_point]]
                seq2 = user_sequences[indices[split_point:]]
                
                test_pairs.append([seq1, seq2])
                labels.append(1)  # Same user
            
            # Create different-user pairs
            other_users = [u for u in users if u != user and len(user_data[u]) >= 50]
            for _ in range(pairs_per_user):
                if not other_users:
                    break
                
                other_user = random.choice(other_users)
                
                # Get random subsequences
                seq1_len = min(50, len(user_sequences))
                seq2_len = min(50, len(user_data[other_user]))
                
                if seq1_len == len(user_sequences):
                    seq1 = user_sequences
                else:
                    start1 = random.randint(0, len(user_sequences) - seq1_len)
                    seq1 = user_sequences[start1:start1 + seq1_len]
                
                if seq2_len == len(user_data[other_user]):
                    seq2 = user_data[other_user]
                else:
                    start2 = random.randint(0, len(user_data[other_user]) - seq2_len)
                    seq2 = user_data[other_user][start2:start2 + seq2_len]
                
                test_pairs.append([seq1, seq2])
                labels.append(0)  # Different users
        
        print(f"✅ Created {len(test_pairs)} pairs ({sum(labels)} same-user, {len(labels) - sum(labels)} different-user)")
        
        if not test_pairs:
            return None, None
        
        # Preprocess pairs (normalize and pad)
        print("🔧 Preprocessing pairs...")
        processed_pairs = self._preprocess_pairs(test_pairs)
        
        return processed_pairs, np.array(labels)
    
    def _preprocess_pairs(self, pairs):
        """Preprocess pairs with normalization and padding (same as training)."""
        # Collect all sequences for normalization
        all_sequences = []
        for pair in pairs:
            for seq in pair:
                # Pad or truncate to fixed length
                if len(seq) < self.sequence_length:
                    padded = np.pad(seq, ((0, self.sequence_length - len(seq)), (0, 0)), 'constant')
                else:
                    padded = seq[:self.sequence_length]
                all_sequences.append(padded)
        
        all_sequences = np.array(all_sequences)
        
        # Normalize each feature separately using RobustScaler
        for feature_idx in range(4):
            scaler = RobustScaler()
            feature_values = all_sequences[:, :, feature_idx].flatten()
            non_zero_mask = feature_values != 0
            
            if np.sum(non_zero_mask) > 0:
                # Fit scaler only on non-zero values
                scaler.fit(feature_values[non_zero_mask].reshape(-1, 1))
                # Transform all values
                transformed = scaler.transform(feature_values.reshape(-1, 1)).flatten()
                all_sequences[:, :, feature_idx] = transformed.reshape(len(all_sequences), self.sequence_length)
        
        # Reconstruct pairs
        processed_pairs = []
        for i in range(0, len(all_sequences), 2):
            processed_pairs.append([all_sequences[i], all_sequences[i+1]])
        
        return np.array(processed_pairs)
    
    def test_on_synthetic_data(self):
        """Test model on synthetic data."""
        print("\n" + "="*60)
        print("🧪 TESTING ON SYNTHETIC DATA")
        print("="*60)
        
        # Load synthetic data
        user_data = self.load_synthetic_data()
        if user_data is None:
            return False
        
        # Create test pairs
        X_synthetic, y_synthetic = self.create_test_pairs_from_synthetic(user_data)
        if X_synthetic is None:
            print("❌ Failed to create test pairs!")
            return False
        
        # Convert to tensors and test
        X_tensor = torch.tensor(X_synthetic, dtype=torch.float32).to(self.device)
        
        print(f"🔮 Testing on {len(X_synthetic)} synthetic pairs...")
        probabilities = []
        
        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[i:i+batch_size]
                seq1 = batch_x[:, 0]
                seq2 = batch_x[:, 1]
                
                batch_pred = self.model(seq1, seq2)
                probabilities.extend(batch_pred.cpu().numpy())
        
        probabilities = np.array(probabilities)
        predictions = (probabilities > 0.5).astype(int)
        
        # Display results
        self._display_metrics(y_synthetic, predictions, probabilities, "SYNTHETIC DATA")
        
        return True
    
    def _display_metrics(self, y_true, predictions, probabilities, data_type):
        """Display comprehensive metrics."""
        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        
        # Handle edge cases for precision/recall
        unique_preds = np.unique(predictions)
        if len(unique_preds) > 1:
            precision = precision_score(y_true, predictions)
            recall = recall_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)
            auc = roc_auc_score(y_true, probabilities)
        else:
            precision = recall = f1 = 0.0
            auc = 0.5  # Random performance if all predictions are the same
        
        cm = confusion_matrix(y_true, predictions)
        
        # Print results
        print(f"\n📊 {data_type} PERFORMANCE:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {auc:.4f}")
        
        print(f"\n🔍 CONFUSION MATRIX:")
        print(f"   True Negatives:  {cm[0,0]} (Different users correctly rejected)")
        print(f"   False Positives: {cm[0,1]} (Different users incorrectly accepted)")
        print(f"   False Negatives: {cm[1,0]} (Same users incorrectly rejected)")
        print(f"   True Positives:  {cm[1,1]} (Same users correctly accepted)")
        
        # Error rates
        far = cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
        frr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        
        print(f"\n🚨 ERROR RATES:")
        print(f"   False Accept Rate (FAR):  {far:.4f} ({far*100:.2f}%)")
        print(f"   False Reject Rate (FRR):  {frr:.4f} ({frr*100:.2f}%)")
        
        # Probability distributions
        same_user_probs = probabilities[y_true == 1]
        diff_user_probs = probabilities[y_true == 0]
        
        print(f"\n📈 PROBABILITY DISTRIBUTIONS:")
        if len(same_user_probs) > 0:
            print(f"   Same User Pairs: μ={same_user_probs.mean():.4f}, σ={same_user_probs.std():.4f}")
        if len(diff_user_probs) > 0:
            print(f"   Different User Pairs: μ={diff_user_probs.mean():.4f}, σ={diff_user_probs.std():.4f}")
        
        # Threshold analysis
        if len(unique_preds) > 1:
            print(f"\n🎯 THRESHOLD ANALYSIS:")
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            for threshold in thresholds:
                thresh_pred = (probabilities > threshold).astype(int)
                thresh_acc = accuracy_score(y_true, thresh_pred)
                thresh_cm = confusion_matrix(y_true, thresh_pred)
                
                if thresh_cm.shape == (2, 2):
                    thresh_far = thresh_cm[0,1] / (thresh_cm[0,1] + thresh_cm[0,0]) if (thresh_cm[0,1] + thresh_cm[0,0]) > 0 else 0
                    thresh_frr = thresh_cm[1,0] / (thresh_cm[1,0] + thresh_cm[1,1]) if (thresh_cm[1,0] + thresh_cm[1,1]) > 0 else 0
                    print(f"   Threshold {threshold:.1f}: Acc={thresh_acc:.3f}, FAR={thresh_far:.3f}, FRR={thresh_frr:.3f}")
    
    def run_comprehensive_test(self):
        """Run complete evaluation on both datasets."""
        print("🔬 COMPREHENSIVE MODEL TESTING")
        print("="*80)
        
        if not self.load_model():
            return
        
        # Test on processed data first
        processed_works = self.test_on_processed_data()
        
        if processed_works:
            print("\n✅ Model works correctly on processed data!")
            
            # Test on synthetic data
            synthetic_works = self.test_on_synthetic_data()
            
            if synthetic_works:
                print("\n🎉 Comprehensive testing completed successfully!")
            else:
                print("\n⚠️ Issues found with synthetic data testing.")
        else:
            print("\n❌ Model has issues with processed data. Check model training.")
        
        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("="*80)

def main():
    """Main function to run testing."""
    tester = ModelTester()
    tester.run_comprehensive_test()

if __name__ == '__main__':
    main() 