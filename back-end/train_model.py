import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import os

class SiameseNetwork(nn.Module):
    """
    A Siamese network with an LSTM base network for keystroke biometric authentication.
    --- MODIFIED ARCHITECTURE ---
    This version uses a more powerful classifier head that combines the embeddings
    instead of relying solely on Euclidean distance.
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout_rate=0.4):
        super(SiameseNetwork, self).__init__()
        # Base network to produce embeddings
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_dim)
        )

        # --- NEW: A more powerful classifier head ---
        # It takes the concatenated embeddings and their difference as input.
        # Input size will be 2 * embedding_dim (for e1 and e2)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        """
        Forward pass for one branch of the Siamese network to get the embedding.
        """
        _, (h_n, _) = self.lstm(x)
        embedding = self.fc(h_n.squeeze(0))
        return embedding

    def forward(self, input1, input2):
        """
        Forward pass for the entire network.
        """
        # Get the embeddings for each input
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)

        # --- MODIFIED: Combine embeddings for the classifier ---
        # Instead of distance, we concatenate the embeddings to give the
        # classifier more information.
        combined_embeddings = torch.cat((embedding1, embedding2), dim=1)
        
        output = self.classifier(combined_embeddings)
        return output

# (The EarlyStopping class remains the same as before)
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='saved_models/siamese_model.pth'):
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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def main():
    """
    Main function to train the Siamese network.
    """
    # --- 1. Load Data ---
    print("Loading data...")
    try:
        X = np.load('processed_data/X_processed.npy')
        y = np.load('processed_data/y_processed.npy')
    except FileNotFoundError:
        print("Error: Make sure 'X_processed.npy' and 'y_processed.npy' are in the 'processed_data' directory.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train1 = torch.tensor(X_train[:, 0, :, :], dtype=torch.float32)
    X_train2 = torch.tensor(X_train[:, 1, :, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val1 = torch.tensor(X_val[:, 0, :, :], dtype=torch.float32)
    X_val2 = torch.tensor(X_val[:, 1, :, :], dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train1, X_train2, y_train_tensor)
    val_dataset = TensorDataset(X_val1, X_val2, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # --- 2. Initialize Model, Loss, and Optimizer ---
    print("Initializing model...")
    input_dim = 4
    hidden_dim = 128
    embedding_dim = 64
    model = SiameseNetwork(input_dim, hidden_dim, embedding_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5) # Slightly reduced learning rate

    # --- 3. Training Loop ---
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    early_stopping = EarlyStopping(patience=10, verbose=True, path='saved_models/siamese_model_best.pth') # Increased patience
    
    print(f"Training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for seq1, seq2, labels in train_loader:
            seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(seq1, seq2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for seq1_val, seq2_val, labels_val in val_loader:
                seq1_val, seq2_val, labels_val = seq1_val.to(device), seq2_val.to(device), labels_val.to(device)
                outputs_val = model(seq1_val, seq2_val)
                loss = criterion(outputs_val, labels_val)
                val_loss += loss.item()
                predicted = (outputs_val > 0.5).float()
                total += labels_val.size(0)
                correct += (predicted == labels_val).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print("\nTraining complete.")
    print(f"Best model saved to {early_stopping.path}")

if __name__ == '__main__':
    main()