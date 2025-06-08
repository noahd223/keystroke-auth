import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# --- Configuration ---
PROCESSED_DATA_FILE = 'processed_data.pt'
TRAINED_MODEL_FILE = 'keystroke_model.pth'

# --- Model Hyperparameters ---
# These are the settings for our neural network. You can tune them later.
INPUT_SIZE = 4      # Number of features per keystroke (dwell, p2p, r2p, r2r)
HIDDEN_SIZE = 64    # Size of the LSTM's memory
NUM_LAYERS = 2      # Number of LSTM layers stacked on top of each other
# NUM_CLASSES will be determined from the data
DROPOUT_RATE = 0.5  # Helps prevent the model from just memorizing the data

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 25     # How many times to show the entire dataset to the model

# --- 1. Model Definition ---
# Here we define the architecture of our neural network.
class KeystrokeAuthenticator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(KeystrokeAuthenticator, self).__init__()
        # The LSTM layer processes the sequence of keystroke timings.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout_rate)
        # The fully connected layer takes the LSTM's output and makes the final prediction.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # We only care about the final output of the LSTM from the last keystroke.
        # h_n contains the hidden state of the last time step.
        _, (h_n, _) = self.lstm(x)
        # We take the output from the last layer's hidden state and pass it to our final layer.
        out = self.fc(h_n[-1, :, :])
        return out

# --- 2. Data Loading and Preparation ---
def load_and_prepare_data():
    """Loads the preprocessed data and splits it into training and validation sets."""
    print("Loading preprocessed data...")
    try:
        data = torch.load(PROCESSED_DATA_FILE)
        X = data['sequences']
        y = data['labels']
        user_to_id = data['user_to_id']
        print(f"Data loaded successfully. Found {len(X)} sequences.")
    except FileNotFoundError:
        print(f"Error: Processed data file '{PROCESSED_DATA_FILE}' not found.")
        print("Please run the 'preprocess_data.py' script first.")
        return None, None, None, None

    # Create a TensorDataset, which is a PyTorch-specific way to handle datasets.
    dataset = TensorDataset(X, y)

    # Split the data into a training set and a validation set (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders to handle batching and shuffling of the data.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    num_classes = len(user_to_id)
    return train_loader, val_loader, num_classes, user_to_id


# --- 3. Training Loop ---
def train_model():
    """The main function to orchestrate the model training process."""
    train_loader, val_loader, num_classes, user_to_id = load_and_prepare_data()

    if train_loader is None:
        return # Stop if data loading failed

    print(f"\nInitializing model for {num_classes} users...")
    
    # Initialize the model, loss function, and optimizer
    model = KeystrokeAuthenticator(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, DROPOUT_RATE)
    criterion = nn.CrossEntropyLoss() # Good for classification problems
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    # Loop over the dataset multiple times (epochs)
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train() # Set the model to training mode
        total_train_loss = 0
        for sequences, labels in train_loader:
            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute gradients
            optimizer.step()      # Update weights
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval() # Set the model to evaluation mode
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad(): # We don't need to compute gradients during validation
            for sequences, labels in val_loader:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_samples) * 100

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # --- 4. Save the Trained Model ---
    print("\nTraining complete.")
    
    # We save the model's learned weights, plus other important info
    model_data = {
        'model_state_dict': model.state_dict(),
        'input_size': INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_classes': num_classes,
        'dropout_rate': DROPOUT_RATE,
        'user_to_id': user_to_id
    }
    torch.save(model_data, TRAINED_MODEL_FILE)
    print(f"Trained model saved to '{TRAINED_MODEL_FILE}'.")


if __name__ == '__main__':
    train_model()
