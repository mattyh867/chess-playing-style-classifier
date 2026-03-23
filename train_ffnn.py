# ffnn_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import gc
from datetime import datetime

from nnprep import (
    load_and_prepare_data,
    create_train_val_test_split,
    normalize_features,
    create_dataloaders,
    save_preprocessing_artifacts
)

gc.enable()

def clear_memory():
    gc.collect()


class ChessStyleFFNN(nn.Module):
    """
    Feedforward neural network for classifying chess playstyles.
    Uses two hidden layers with dropout for regularisation.
    """
    def __init__(self, input_size=6, hidden1_size=64, hidden2_size=32,
                 num_classes=4, dropout_rate=0.3):
        super(ChessStyleFFNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

def main():
    CSV_PATH = 'labelling/labeled_dataset_2015.csv'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    RANDOM_STATE = 42
    PATIENCE = 15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nFEEDFORWARD NEURAL NETWORK TRAINING")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {NUM_EPOCHS}")

    clear_memory()

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

    clear_memory()

    print("\nInitializing FFNN model...")
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

    print("\nTraining model...")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {PATIENCE} epochs)")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break

    model.load_state_dict(best_model_state)
    print("\nTraining complete")

    clear_memory()

    print("\nEvaluating on test set...")

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

    label_names = list(label_mapping.keys())

    print("\nCLASSIFICATION REPORT")
    print(classification_report(all_labels, all_predictions,
                                target_names=label_names, digits=3))

    print("\nCONFUSION MATRIX")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )

    metrics_df = pd.DataFrame({
        'Class': label_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    print("\nPER-CLASS METRICS")
    print(metrics_df)

    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nOverall Accuracy: {test_accuracy:.4f}")

    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Confusion Matrix - FFNN', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('models/FFNN/confusion_matrix_ffnn.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to models/FFNN/confusion_matrix_ffnn.png")
    plt.close()

    os.makedirs('models/FFNN', exist_ok=True)

    feature_cols_path = 'models/FFNN/feature_columns.pkl'
    joblib.dump(feature_columns, feature_cols_path)
    print(f"Feature columns saved: {feature_cols_path}")

    scaler_path = 'models/FFNN/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    model_path = 'models/FFNN/ffnn_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    history_path = 'models/FFNN/training_history.pkl'
    joblib.dump(history, history_path)
    print(f"History saved: {history_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        'test_accuracy': test_accuracy,
        'timestamp': timestamp,
        'architecture': {
            'input_size': len(feature_columns),
            'hidden1_size': 64,
            'hidden2_size': 32,
            'num_classes': 4,
            'dropout_rate': 0.3
        }
    }

    metadata_path = 'models/FFNN/ffnn_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"Metadata saved: {metadata_path}")

    results = {
        'accuracy': test_accuracy,
        'per_class_metrics': metrics_df.to_dict('records'),
        'confusion_matrix': cm.tolist(),
    }

    with open('results/FFNN/evaluation_metrics_ffnn.json', 'w') as f:
        json.dump(results, f, indent=2)

    clear_memory()

    print("\nSUMMARY")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_accuracy*100:.2f}%")
    print(f"Training epochs: {len(history['train_loss'])}")
    print(f"\nTRAINING COMPLETE")

if __name__ == "__main__":
    main()