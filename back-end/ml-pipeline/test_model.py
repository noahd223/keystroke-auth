import torch
import torch.nn as nn
import numpy as np
from train_model_improved import ImprovedSiameseNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

def load_test_data():
    """Load the processed test data."""
    try:
        # Load processed data
        X = np.load('processed_data/X_processed.npy')
        y = np.load('processed_data/y_processed.npy')
        
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
    model.load_state_dict(torch.load('saved_models/improved_siamese_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model, device

def evaluate_model():
    """Comprehensive model evaluation."""
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
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
    print(f"\n📊 TEST SET PERFORMANCE:")
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
    
    # Calculate error rates
    far = cm[0,1] / (cm[0,1] + cm[0,0])  # False Accept Rate
    frr = cm[1,0] / (cm[1,0] + cm[1,1])  # False Reject Rate
    
    print(f"\n🚨 ERROR RATES:")
    print(f"   False Accept Rate (FAR):  {far:.4f} ({far*100:.2f}%)")
    print(f"   False Reject Rate (FRR):  {frr:.4f} ({frr*100:.2f}%)")
    
    # Probability distribution analysis
    same_user_probs = probabilities[y_test == 1]
    diff_user_probs = probabilities[y_test == 0]
    
    print(f"\n📈 PROBABILITY DISTRIBUTIONS:")
    print(f"   Same User Pairs:")
    print(f"     Mean: {same_user_probs.mean():.4f}")
    print(f"     Std:  {same_user_probs.std():.4f}")
    print(f"     Min:  {same_user_probs.min():.4f}")
    print(f"     Max:  {same_user_probs.max():.4f}")
    
    print(f"   Different User Pairs:")
    print(f"     Mean: {diff_user_probs.mean():.4f}")
    print(f"     Std:  {diff_user_probs.std():.4f}")
    print(f"     Min:  {diff_user_probs.min():.4f}")
    print(f"     Max:  {diff_user_probs.max():.4f}")
    
    # Threshold analysis
    print(f"\n🎯 THRESHOLD ANALYSIS:")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        thresh_pred = (probabilities > threshold).astype(int)
        thresh_acc = accuracy_score(y_test, thresh_pred)
        thresh_cm = confusion_matrix(y_test, thresh_pred)
        thresh_far = thresh_cm[0,1] / (thresh_cm[0,1] + thresh_cm[0,0]) if (thresh_cm[0,1] + thresh_cm[0,0]) > 0 else 0
        thresh_frr = thresh_cm[1,0] / (thresh_cm[1,0] + thresh_cm[1,1]) if (thresh_cm[1,0] + thresh_cm[1,1]) > 0 else 0
        
        print(f"   Threshold {threshold:.1f}: Acc={thresh_acc:.3f}, FAR={thresh_far:.3f}, FRR={thresh_frr:.3f}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'far': far,
        'frr': frr,
        'probabilities': probabilities,
        'predictions': predictions,
        'y_test': y_test
    }

if __name__ == "__main__":
    results = evaluate_model() 