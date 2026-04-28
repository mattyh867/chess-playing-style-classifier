import pandas as pd
import numpy as np
import joblib
import json
import os
import gc
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from ffnn import ChessStyleFFNN

gc.enable()


def clear_memory():
    gc.collect()


RF_MODEL_PATH      = 'models/RF/rf_baseline_model.pkl'
RF_SCALER_PATH     = 'models/RF/scaler.pkl'
RF_FEATURES_PATH   = 'models/RF/feature_columns.pkl'

XGB_MODEL_PATH     = 'models/XGB/xgb_model.pkl'
XGB_SCALER_PATH    = 'models/XGB/scaler.pkl'
XGB_FEATURES_PATH  = 'models/XGB/feature_columns.pkl'
XGB_ENCODER_PATH   = 'models/XGB/label_encoder.pkl'

FFNN_MODEL_PATH    = 'models/FFNN/ffnn_model.pth'
FFNN_SCALER_PATH   = 'models/FFNN/scaler.pkl'
FFNN_FEATURES_PATH = 'models/FFNN/feature_columns.pkl'

HIGH_ELO_CSV       = 'tests/eval_high.csv'
LOW_ELO_CSV        = 'tests/eval_low.csv'

CLASSES = ['aggressive', 'positional', 'defensive', 'balanced']

MODEL_COLORS = {
    'Random Forest': '#67000D',
    'FFNN':          '#08306B',
    'XGBoost':       '#00441B',
}


def load_rf():
    model        = joblib.load(RF_MODEL_PATH)
    scaler       = joblib.load(RF_SCALER_PATH)
    feature_cols = joblib.load(RF_FEATURES_PATH)
    return model, scaler, feature_cols


def load_xgb():
    model        = joblib.load(XGB_MODEL_PATH)
    scaler       = joblib.load(XGB_SCALER_PATH)
    feature_cols = joblib.load(XGB_FEATURES_PATH)
    encoder      = joblib.load(XGB_ENCODER_PATH)
    return model, scaler, feature_cols, encoder


def load_ffnn():
    scaler       = joblib.load(FFNN_SCALER_PATH)
    feature_cols = joblib.load(FFNN_FEATURES_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChessStyleFFNN(
        input_size=len(feature_cols),
        hidden1_size=64,
        hidden2_size=32,
        num_classes=4,
        dropout_rate=0.3
    ).to(device)
    model.load_state_dict(torch.load(FFNN_MODEL_PATH, map_location=device))
    model.eval()
    return model, scaler, feature_cols, device


def prepare_features(df, feature_cols):
    metadata_cols = [
        'player_name', 'player_elo', 'color', 'game_id',
        'date', 'result', 'opening', 'label',
        'king_safety_risks', 'complex_positions'
    ]
    available = [col for col in feature_cols if col in df.columns]
    missing   = [col for col in feature_cols if col not in df.columns]

    if missing:
        print(f"Warning: missing features: {missing}")

    return df[available]


def predict_rf(df, model, scaler, feature_cols):
    X = prepare_features(df, feature_cols)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def predict_xgb(df, model, scaler, feature_cols, encoder):
    X = prepare_features(df, feature_cols)
    X_scaled = scaler.transform(X)
    y_encoded = model.predict(X_scaled)
    return encoder.inverse_transform(y_encoded)


def predict_ffnn(df, model, scaler, feature_cols, device):
    X = prepare_features(df, feature_cols)
    X_scaled = scaler.transform(X)
    tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    label_map = {0: 'aggressive', 1: 'positional', 2: 'defensive', 3: 'balanced'}
    return np.array([label_map[i] for i in predicted.cpu().numpy()])


def print_distribution(predictions, group_name, model_name):
    print(f"\n{model_name} - {group_name}")
    print("-" * 40)
    series      = pd.Series(predictions)
    counts      = series.value_counts()
    percentages = series.value_counts(normalize=True) * 100
    for cls in CLASSES:
        count = counts.get(cls, 0)
        pct   = percentages.get(cls, 0.0)
        print(f"  {cls:<12} {count:>5}  ({pct:.1f}%)")


def save_distribution_chart(predictions_dict, group_name):
    plt.figure(figsize=(10, 8))

    x     = np.arange(len(CLASSES))
    width = 0.25

    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        series = pd.Series(preds)
        percentages = [
            series.value_counts(normalize=True).get(cls, 0) * 100
            for cls in CLASSES
        ]
        plt.bar(
            x + i * width,
            percentages,
            width,
            label=model_name,
            color=MODEL_COLORS[model_name]
        )

    plt.xlabel('Playing Style', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title(
        f'Predicted Class Distribution - {group_name}',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.xticks(x + width, CLASSES)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    filename_map = {
        'High Elo (2000+)':       'distribution_high_elo.png',
        'Low Elo (under 1400)':   'distribution_low_elo.png',
    }
    save_path = f'results/evaluation/{filename_map[group_name]}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved: {save_path}")
    plt.close()


def save_results_json(all_results):
    path = 'results/evaluation/elo_evaluation_results.json'
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {path}")


def main():
    print("\nELO GROUP EVALUATION")

    clear_memory()

    print("\nLoading models...")
    rf_model,   rf_scaler,   rf_features              = load_rf()
    xgb_model,  xgb_scaler,  xgb_features,  xgb_enc  = load_xgb()
    ffnn_model, ffnn_scaler, ffnn_features,  device   = load_ffnn()
    print("All models loaded")

    groups = {}

    if os.path.exists(HIGH_ELO_CSV):
        groups['High Elo (2000+)'] = pd.read_csv(HIGH_ELO_CSV)
        print(f"\nHigh elo dataset loaded: {len(groups['High Elo (2000+)'])} samples")
    else:
        print(f"\nHigh elo CSV not found at {HIGH_ELO_CSV} - skipping")

    if os.path.exists(LOW_ELO_CSV):
        groups['Low Elo (under 1400)'] = pd.read_csv(LOW_ELO_CSV)
        print(f"Low elo dataset loaded: {len(groups['Low Elo (under 1400)'])} samples")
    else:
        print(f"Low elo CSV not found at {LOW_ELO_CSV} - skipping")

    if not groups:
        print("\nNo evaluation CSVs found. Run processing.py first.")
        return

    os.makedirs('results/evaluation', exist_ok=True)

    all_results = {}

    for group_name, df in groups.items():
        print(f"\nGROUP: {group_name}")
        print(f"Samples: {len(df)}")
        print(f"Elo range: {df['player_elo'].min()} - {df['player_elo'].max()}")

        rf_preds   = predict_rf(df, rf_model, rf_scaler, rf_features)
        xgb_preds  = predict_xgb(df, xgb_model, xgb_scaler, xgb_features, xgb_enc)
        ffnn_preds = predict_ffnn(df, ffnn_model, ffnn_scaler, ffnn_features, device)

        clear_memory()

        predictions_dict = {
            'Random Forest': rf_preds,
            'FFNN':          ffnn_preds,
            'XGBoost':       xgb_preds,
        }

        for model_name, preds in predictions_dict.items():
            print_distribution(preds, group_name, model_name)

        save_distribution_chart(predictions_dict, group_name)

        agreement = np.mean(
            (rf_preds == xgb_preds) & (xgb_preds == ffnn_preds)
        ) * 100
        print(f"\nModel agreement (all three match): {agreement:.1f}%")

        all_results[group_name] = {
            'n_samples': len(df),
            'elo_range': [int(df['player_elo'].min()), int(df['player_elo'].max())],
            'model_agreement_pct': round(agreement, 2),
            'Random Forest': {
                cls: round(float((pd.Series(rf_preds) == cls).mean() * 100), 2)
                for cls in CLASSES
            },
            'FFNN': {
                cls: round(float((pd.Series(ffnn_preds) == cls).mean() * 100), 2)
                for cls in CLASSES
            },
            'XGBoost': {
                cls: round(float((pd.Series(xgb_preds) == cls).mean() * 100), 2)
                for cls in CLASSES
            },
        }

    save_results_json(all_results)

    print("\nEVALUATION COMPLETE")

if __name__ == "__main__":
    main()