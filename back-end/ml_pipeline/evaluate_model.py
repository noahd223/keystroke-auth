import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .train_model import ImprovedSiameseNetwork
from sklearn.preprocessing import RobustScaler
import os

class KeystrokeAuthenticator:
    """
    A complete keystroke authentication system that can authenticate users
    based on their typing patterns.
    """
    
    def __init__(self, model_path='saved_models/improved_siamese_model.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 50
        self.features = ['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time']
        self.scalers = {}
        self.enrolled_users = {}
        
    def load_model(self):
        """Load the trained model."""
        try:
            # Initialize model architecture
            input_dim = 4  # Number of features
            hidden_dim = 64
            embedding_dim = 32
            
            self.model = ImprovedSiameseNetwork(input_dim, hidden_dim, embedding_dim)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_sequence(self, keystroke_data):
        """
        Preprocess a single keystroke sequence for authentication.
        """
        # Convert to DataFrame if it's not already
        if isinstance(keystroke_data, list):
            df = pd.DataFrame(keystroke_data)
        else:
            df = keystroke_data.copy()
        
        # Extract features
        if not all(feature in df.columns for feature in self.features):
            raise ValueError(f"Missing required features. Need: {self.features}")
        
        sequence = df[self.features].values
        
        # Remove outliers
        sequence = sequence[(sequence > 0).all(axis=1)]
        sequence = sequence[(sequence < 5000).all(axis=1)]
        
        if len(sequence) == 0:
            raise ValueError("No valid keystrokes after preprocessing")
        
        # Pad or truncate to fixed length
        if len(sequence) < self.sequence_length:
            pad_width = self.sequence_length - len(sequence)
            sequence = np.pad(sequence, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
        else:
            sequence = sequence[:self.sequence_length]
        
        # Normalize features separately
        for i, feature in enumerate(self.features):
            if feature in self.scalers:
                # Use existing scaler
                scaler = self.scalers[feature]
                feature_values = sequence[:, i].flatten()
                non_zero_mask = feature_values != 0
                if np.sum(non_zero_mask) > 0:
                    transformed = scaler.transform(feature_values[non_zero_mask].reshape(-1, 1)).flatten()
                    sequence[non_zero_mask, i] = transformed
            else:
                # Create new scaler for this feature
                scaler = RobustScaler()
                feature_values = sequence[:, i].flatten()
                non_zero_mask = feature_values != 0
                if np.sum(non_zero_mask) > 0:
                    scaler.fit(feature_values[non_zero_mask].reshape(-1, 1))
                    transformed = scaler.transform(feature_values[non_zero_mask].reshape(-1, 1)).flatten()
                    sequence[non_zero_mask, i] = transformed
                    self.scalers[feature] = scaler
        
        return sequence
    
    def enroll_user(self, username, keystroke_sequences):
        """
        Enroll a user by storing their reference keystroke patterns.
        
        Args:
            username: The username to enroll
            keystroke_sequences: List of keystroke sequences for this user
        """
        try:
            processed_sequences = []
            for seq in keystroke_sequences:
                processed_seq = self.preprocess_sequence(seq)
                processed_sequences.append(processed_seq)
            
            self.enrolled_users[username] = processed_sequences
            print(f"User '{username}' enrolled with {len(processed_sequences)} reference sequences")
            return True
        except Exception as e:
            print(f"Error enrolling user {username}: {e}")
            return False
    
    def authenticate_user(self, username, test_sequence, threshold=0.5):
        """
        Authenticate a user based on their keystroke pattern.
        
        Args:
            username: The claimed username
            test_sequence: The keystroke sequence to authenticate
            threshold: Authentication threshold (default 0.5)
            
        Returns:
            dict: Authentication result with probability scores
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if username not in self.enrolled_users:
            return {
                'authenticated': False,
                'probability': 0.0,
                'error': f"User '{username}' not enrolled"
            }
        
        try:
            # Preprocess test sequence
            test_seq_processed = self.preprocess_sequence(test_sequence)
            
            # Compare with all enrolled sequences for this user
            probabilities = []
            
            for ref_sequence in self.enrolled_users[username]:
                # Convert to tensors
                seq1_tensor = torch.tensor(ref_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                seq2_tensor = torch.tensor(test_seq_processed, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    probability = self.model(seq1_tensor, seq2_tensor).item()
                    probabilities.append(probability)
            
            # Use the maximum probability (best match)
            max_probability = max(probabilities)
            avg_probability = np.mean(probabilities)
            
            # Authentication decision
            authenticated = max_probability >= threshold
            
            return {
                'authenticated': authenticated,
                'max_probability': max_probability,
                'avg_probability': avg_probability,
                'all_probabilities': probabilities,
                'threshold': threshold,
                'username': username
            }
            
        except Exception as e:
            return {
                'authenticated': False,
                'probability': 0.0,
                'error': f"Authentication error: {e}"
            }
    
    def authenticate_against_all_users(self, test_sequence, threshold=0.5):
        """
        Test authentication against all enrolled users to find the best match.
        
        Args:
            test_sequence: The keystroke sequence to authenticate
            threshold: Authentication threshold
            
        Returns:
            dict: Results for all users with the best match identified
        """
        if not self.enrolled_users:
            return {'error': 'No users enrolled'}
        
        results = {}
        best_match = None
        best_probability = 0.0
        
        for username in self.enrolled_users:
            result = self.authenticate_user(username, test_sequence, threshold)
            results[username] = result
            
            if 'max_probability' in result and result['max_probability'] > best_probability:
                best_probability = result['max_probability']
                best_match = username
        
        return {
            'best_match': best_match,
            'best_probability': best_probability,
            'all_results': results,
            'authenticated': best_probability >= threshold
        }

def demo_authentication():
    """
    Demonstration of how to use the keystroke authenticator.
    """
    print("Keystroke Authentication Demo")
    print("=" * 40)
    
    # Initialize authenticator
    authenticator = KeystrokeAuthenticator()
    
    # Load model
    if not authenticator.load_model():
        print("Failed to load model. Make sure you've trained the model first.")
        return
    
    print("\nDemo: Load some sample data and demonstrate authentication")
    
    # Try to load some sample data
    try:
        # Load processed data to get some sample sequences
        X = np.load('processed_data/X_processed.npy')
        y = np.load('processed_data/y_processed.npy')
        
        print(f"Loaded data with shape: {X.shape}")
        
        # Take a few sample sequences for demonstration
        sample_seq1 = X[0, 0, :, :]  # First sequence from first pair
        sample_seq2 = X[0, 1, :, :]  # Second sequence from first pair
        sample_seq3 = X[1, 0, :, :]  # First sequence from second pair
        
        # Convert back to DataFrame format (approximate)
        def array_to_keystroke_df(arr):
            return pd.DataFrame(arr, columns=['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time'])
        
        df1 = array_to_keystroke_df(sample_seq1)
        df2 = array_to_keystroke_df(sample_seq2)
        df3 = array_to_keystroke_df(sample_seq3)
        
        # Enroll a user with reference sequences
        print("\nEnrolling user 'demo_user'...")
        authenticator.enroll_user('demo_user', [df1, df2])
        
        # Test authentication with a sequence from the same user
        print("\nTesting authentication with same user's sequence...")
        result1 = authenticator.authenticate_user('demo_user', df2)
        print(f"Result: {result1}")
        
        # Test authentication with a different user's sequence
        print("\nTesting authentication with different user's sequence...")
        result2 = authenticator.authenticate_user('demo_user', df3)
        print(f"Result: {result2}")
        
        print("\nDemo completed successfully!")
        
    except FileNotFoundError:
        print("No processed data found. Please run preprocessing first.")
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == '__main__':
    demo_authentication() 