import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
import json
import os

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    exit(1)

gc.enable()

def clear_memory():
    gc.collect()


def main():
    clear_memory()

    df = pd.read_csv("labelling/labeled_dataset_2017.csv")

    print(f"Dataset loaded: {len(df)} games")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())

    metadata_cols = [
        'player_name', 'player_elo', 'color', 'game_id',
        'date', 'result', 'opening', 'label'
    ]

    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=42
    )

    clear_memory()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # XGBoost requires numeric labels
    classes = ['aggressive', 'positional', 'defensive', 'balanced']
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    sample_weights = compute_sample_weight('balanced', y_train_encoded)

    print("\nInitializing XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=1
    )

    print("\nTraining model...")
    model.fit(
        X_train_scaled, y_train_encoded,
        eval_set=[(X_val_scaled, y_val_encoded)],
        verbose=False,
        sample_weight=sample_weights,
    )

    clear_memory()

    y_pred_encoded = model.predict(X_test_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, labels=classes))

    print("\nCONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(cm)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes
    )

    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    print("\nPER-CLASS METRICS")
    print(metrics_df)
    print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTOP 15 MOST IMPORTANT FEATURES")
    print(feature_importance.head(15).to_string(index=False))

    os.makedirs('models/XGB', exist_ok=True)
    os.makedirs('results/XGB', exist_ok=True)

    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Greens',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('models/XGB/confusion_matrix_xgb.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to models/XGB/confusion_matrix_xgb.png")
    plt.close()

    joblib.dump(model, 'models/XGB/xgb_model.pkl')
    joblib.dump(scaler, 'models/XGB/scaler.pkl')
    joblib.dump(feature_cols, 'models/XGB/feature_columns.pkl')
    joblib.dump(label_encoder, 'models/XGB/label_encoder.pkl')

    feature_importance.to_csv('results/XGB/feature_importance_xgb.csv', index=False)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'per_class_metrics': metrics_df.to_dict('records'),
        'confusion_matrix': cm.tolist(),
        'feature_importance_top15': feature_importance.head(15).to_dict('records')
    }

    with open('results/XGB/evaluation_metrics_xgb.json', 'w') as f:
        json.dump(results, f, indent=2)

    clear_memory()

    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()