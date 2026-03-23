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
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
import json
import os
from datetime import datetime

try:
    from xgboost import XGBClassifier
    print("XGBoost imported successfully")
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    exit(1)

gc.enable()

def clear_memory():
    gc.collect()

CSV_PATH = 'labelling/labeled_dataset_2015.csv'
RANDOM_STATE = 42
CLASSES = ['aggressive', 'positional', 'defensive', 'balanced']

print("\nXGBOOST CLASSIFIER TRAINING")

clear_memory()

df = pd.read_csv(CSV_PATH)

print(f"\nDataset loaded: {len(df)} games")
print(f"\nClass distribution:")
print(df['label'].value_counts())

metadata_cols = [
    'player_name', 'player_elo', 'color', 'game_id',
    'date', 'result', 'opening', 'label'
]

feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols]
y = df['label']

print(f"\nFeatures: {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\nTrain set: {len(X_train)} games")
print(f"Test set: {len(X_test)} games")

clear_memory()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost requires numeric labels
label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"\nLabel encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

print("\nInitializing XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss',
    verbosity=1
)

print("\nTraining model...")
model.fit(
    X_train_scaled, y_train_encoded,
    eval_set=[(X_test_scaled, y_test_encoded)],
    verbose=False
)

print("Training complete")

clear_memory()

print("\nEVALUATION")

y_pred_encoded = model.predict(X_test_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, labels=CLASSES))

print("\nCONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
print(cm)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=CLASSES
)

metrics_df = pd.DataFrame({
    'Class': CLASSES,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print("\nPER-CLASS METRICS")
print(metrics_df)

xgb_accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {xgb_accuracy:.4f}")

print("\nFEATURE IMPORTANCE")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 15 MOST IMPORTANT FEATURES")
print(feature_importance.head(15).to_string(index=False))

print(f"  {'XGBoost':<20} {f'{xgb_accuracy*100:.1f}%':>10}")

print("\nSAVING RESULTS")

os.makedirs('models/XGB', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Save model
joblib.dump(model, 'models/XGB/xgb_model.pkl')
joblib.dump(scaler, 'models/XGB/scaler.pkl')
joblib.dump(feature_cols, 'models/XGB/feature_columns.pkl')
joblib.dump(label_encoder, 'models/XGB/label_encoder.pkl')
print("Model saved to models/XGB/")

# Save feature importance
feature_importance.to_csv('results/feature_importance_xgb.csv', index=False)
print("Feature importance saved to results/feature_importance_xgb.csv")

# Save confusion matrix plot
plt.figure(figsize=(10, 8))
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
annotations = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Greens',
            xticklabels=CLASSES, yticklabels=CLASSES,
            cbar_kws={'label': 'Percentage (%)'})
plt.title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('models/confusion_matrix_xgb.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved to models/confusion_matrix_xgb.png")
plt.close()

# Save evaluation metrics
results = {
    'accuracy': xgb_accuracy,
    'per_class_metrics': metrics_df.to_dict('records'),
    'confusion_matrix': cm.tolist(),
    'feature_importance_top15': feature_importance.head(15).to_dict('records'),
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
    }
}

with open('results/evaluation_metrics_xgb.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Metrics saved to results/evaluation_metrics_xgb.json")

clear_memory()

print("\nSUMMARY")

print(f"XGBoost Accuracy: {xgb_accuracy*100:.1f}%")
print(f"Top 3 features: {', '.join(feature_importance.head(3)['feature'].tolist())}")
