import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_prepare_data(csv_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print(f"Dataset loaded: {len(df)} games")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())

    metadata_cols = [
        'player_name', 'player_elo', 'color', 'game_id', 
        'date', 'result', 'opening', 'label'
    ]

    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    label_mapping = {
        'aggressive': 0,
        'positional': 1,
        'defensive': 2,
        'balanced': 3
    }
    
    if y.dtype == 'object':  # If labels are strings
        y = np.array([label_mapping[label] for label in y])
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_cols, label_mapping


def create_train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Create train/validation/test splits"""
    print("\nCreating train/val/test splits...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train, X_val, X_test):
    print("\nNormalizing features...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Normalization complete")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """Convert numpy arrays to PyTorch DataLoaders"""
    print(f"\nCreating DataLoaders with batch_size={batch_size}...")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully")
    
    return train_loader, val_loader, test_loader


def save_preprocessing_artifacts(scaler, label_mapping, feature_cols, save_dir='models'):
    """Save scaler and metadata for later use"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving preprocessing artifacts to {save_dir}/...")
    
    # Save scaler
    with open(f'{save_dir}/scaler.pkl', 'wb') as f:
        joblib.dump(scaler, f'{save_dir}/scaler.pkl')
    
    # Save metadata
    metadata = {
        'label_mapping': label_mapping,
        'feature_cols': feature_cols,
        'num_features': len(feature_cols)
    }
    
    with open(f'{save_dir}/metadata.pkl', 'wb') as f:
        joblib.dump(metadata, f'{save_dir}/metadata.pkl')
    
    print("Preprocessing artifacts saved")

if __name__ == "__main__":
    # Configuration
    CSV_PATH = 'labelling/labeled_dataset_2015.csv'
    BATCH_SIZE = 32
    RANDOM_STATE = 42
    
    print("\nNEURAL NETWORK DATA PREPARATION")
    
    # Load data
    X, y, feature_cols, label_mapping = load_and_prepare_data(CSV_PATH)
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X, y, random_state=RANDOM_STATE
    )
    
    # Normalize features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_features(
        X_train, X_val, X_test
    )
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        batch_size=BATCH_SIZE
    )
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(scaler, label_mapping, feature_cols)
    
    print("\nDATA PREPARATION COMPLETE")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Number of classes: {len(label_mapping)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")