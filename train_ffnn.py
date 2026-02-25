import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

from nnprep import (
    load_and_prepare_data,
    create_train_val_test_split,
    normalize_features,
    create_dataloaders,
    save_preprocessing_artifacts
)

class ChessStyleFFNN(nn.Module):
    """
    Feedforward neural network for classifying chess playstyles.
    Uses two hidden layers with dropout for regularisation.
    """ 
    def __init__(self, input_size=6, hidden1_size=64, hidden2_size=32, 
                 num_classes=4, dropout_rate=0.3):
        super(ChessStyleFFNN, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        
    def forward(self, x):
        # Pass through first layer
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Pass through second layer
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output layer (no activation, handled by CrossEntropyLoss)
        x = self.fc3(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch and return loss/accuracy"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Run validation and return loss/accuracy"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, patience=10):
    """
    Main training loop with early stopping.
    """
    print("\n" + "="*60)
    print("Starting training")
    print("="*60)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"  -> New best validation accuracy: {best_val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    print("\n" + "="*60)
    print("Training complete")
    print("="*60)
    
    return model, history, best_val_acc

def evaluate_model(model, test_loader, device, label_mapping):
    """Evaluate model on test set and print detailed metrics"""
    print("\n" + "="*60)
    print("Evaluating on test set")
    print("="*60)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Per-class metrics
    label_names = list(label_mapping.keys())
    print("\n" + "-"*60)
    print("Per-class metrics:")
    print("-"*60)
    print(classification_report(all_labels, all_predictions, 
                                target_names=label_names, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, all_predictions, all_labels, cm

def save_model(model, history, accuracy, save_dir='models'):
    """Save the trained model and metadata"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model weights
    model_path = f'{save_dir}/ffnn_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save training history
    history_path = f'{save_dir}/training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"History saved: {history_path}")
    
    # Save metadata
    metadata = {
        'test_accuracy': accuracy,
        'timestamp': timestamp,
        'architecture': {
            'input_size': 6,
            'hidden1_size': 64,
            'hidden2_size': 32,
            'num_classes': 4,
            'dropout_rate': 0.3
        }
    }
    
    metadata_path = f'{save_dir}/ffnn_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved: {metadata_path}")

def main():
    # Setup
    CSV_PATH = 'labelling/labeled_dataset_2015.csv'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    RANDOM_STATE = 42
    PATIENCE = 15
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("FEEDFORWARD NEURAL NETWORK TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    
    # Load and prepare data
    print("\n" + "="*60)
    print("Loading data")
    print("="*60)
    
    X, y, feature_columns, label_mapping = load_and_prepare_data(CSV_PATH)
    
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X, y, random_state=RANDOM_STATE
    )
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_features(
        X_train, X_val, X_test
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        batch_size=BATCH_SIZE
    )
    
    save_preprocessing_artifacts(scaler, label_mapping, feature_columns)
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing model")
    print("="*60)
    
    model = ChessStyleFFNN(
        input_size=len(feature_columns),
        hidden1_size=64,
        hidden2_size=32,
        num_classes=len(label_mapping),
        dropout_rate=0.3
    ).to(device)
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    model, history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        NUM_EPOCHS, device, patience=PATIENCE
    )
    
    # Evaluate
    test_accuracy, predictions, true_labels, cm = evaluate_model(
        model, test_loader, device, label_mapping
    )
    
    # Save everything
    print("\n" + "="*60)
    print("Saving results")
    print("="*60)
    
    plot_training_history(history)
    
    label_names = list(label_mapping.keys())
    plot_confusion_matrix(cm, label_names)
    
    save_model(model, history, test_accuracy)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_accuracy*100:.2f}%")
    print(f"Training epochs: {len(history['train_loss'])}")
    print("\nComparison to Random Forest baseline:")
    print(f"Random Forest: 83.3%")
    print(f"FFNN: {test_accuracy*100:.2f}%")
    
    print("\n" + "="*60)
    print("Done")
    print("="*60)


if __name__ == "__main__":
    main()