import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
import matplotlib.pyplot as plt

class ImprovedSiameseNetwork(nn.Module):
    """
    Improved Siamese network with better architecture for keystroke authentication.
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout_rate=0.3):
        super(ImprovedSiameseNetwork, self).__init__()
        
        # Improved LSTM base network
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=2,  # Deeper LSTM
            batch_first=True, 
            dropout=dropout_rate,
            bidirectional=True  # Bidirectional for better context
        )
        
        # Attention mechanism to focus on important parts of the sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 because bidirectional
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_dim)
        )
        
        # Improved classifier with more sophisticated distance metrics
        # Input is: concatenation (2*embedding_dim) + difference (embedding_dim) + element-wise product (embedding_dim) = 4*embedding_dim
        classifier_input_dim = embedding_dim * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        """
        Forward pass for one branch with attention mechanism.
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling across time dimension
        pooled = torch.mean(attended_out, dim=1)
        
        # Feature extraction
        embedding = self.feature_extractor(pooled)
        
        return embedding

    def forward(self, input1, input2):
        """
        Forward pass with multiple distance metrics.
        """
        # Get embeddings
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)
        
        # Multiple distance metrics
        concatenated = torch.cat((embedding1, embedding2), dim=1)
        difference = torch.abs(embedding1 - embedding2)
        element_wise_product = embedding1 * embedding2
        
        # Combine all metrics - now this is 4 * embedding_dim
        combined_features = torch.cat((concatenated, difference, element_wise_product), dim=1)
        
        output = self.classifier(combined_features)
        return output

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='saved_models/improved_siamese_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def evaluate_model(model, dataloader, criterion, device):
    """
    Comprehensive model evaluation.
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for seq1, seq2, labels in dataloader:
            seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
            outputs = model(seq1, seq2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    try:
        auc_score = roc_auc_score(all_labels, all_probabilities)
    except:
        auc_score = 0.5
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1, auc_score

def main():
    """
    Improved training function with better monitoring and techniques.
    """
    print("Loading data...")
    try:
        X = np.load('processed_data/X_processed.npy')
        y = np.load('processed_data/y_processed.npy')
        print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
    except FileNotFoundError:
        print("Error: Run preprocessing first to generate processed data files.")
        return

    # Check data quality
    if np.isnan(X).any() or np.isinf(X).any():
        print("Error: Data contains NaN or infinite values")
        return
    
    if len(np.unique(y)) != 2:
        print("Error: Labels should be binary (0 and 1)")
        return

    # Train-test split with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to tensors
    X_train1 = torch.tensor(X_train[:, 0, :, :], dtype=torch.float32)
    X_train2 = torch.tensor(X_train[:, 1, :, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val1 = torch.tensor(X_val[:, 0, :, :], dtype=torch.float32)
    X_val2 = torch.tensor(X_val[:, 1, :, :], dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train1, X_train2, y_train_tensor)
    val_dataset = TensorDataset(X_val1, X_val2, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    print("Initializing improved model...")
    input_dim = X.shape[-1]  # Number of features
    hidden_dim = 64
    embedding_dim = 32
    
    model = ImprovedSiameseNetwork(input_dim, hidden_dim, embedding_dim, dropout_rate=0.3)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}")

    # Loss and optimizer with improved settings
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_aucs = []

    print("Starting training...")
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for seq1, seq2, labels in train_loader:
            seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(seq1, seq2)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        print(f"  Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print("\nTraining complete!")
    print(f"Best model saved to {early_stopping.path}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(val_aucs, label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == '__main__':
    main() 