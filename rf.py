import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
import json
import os

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

    clear_memory()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nInitializing Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        class_weight='balanced'
    )

    print("\nTraining model...")
    model.fit(X_train_scaled, y_train)

    clear_memory()

    y_pred = model.predict(X_test_scaled)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))

    print("\nCONFUSION MATRIX")
    classes = ['aggressive', 'positional', 'defensive', 'balanced']
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

    os.makedirs('models/RF', exist_ok=True)
    os.makedirs('results/RF', exist_ok=True)

    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Reds',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('models/RF/confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to models/RF/confusion_matrix_rf.png")
    plt.close()

    joblib.dump(model, 'models/RF/rf_baseline_model.pkl')
    joblib.dump(scaler, 'models/RF/scaler.pkl')
    joblib.dump(feature_cols, 'models/RF/feature_columns.pkl')

    feature_importance.to_csv('results/RF/feature_importance_rf.csv', index=False)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'per_class_metrics': metrics_df.to_dict('records'),
        'confusion_matrix': cm.tolist(),
        'feature_importance_top15': feature_importance.head(15).to_dict('records')
    }

    with open('results/RF/evaluation_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    clear_memory()

    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()