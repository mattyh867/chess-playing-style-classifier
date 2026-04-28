import os
import io
import joblib
import numpy as np
import chess
import chess.pgn
import torch
from flask import Flask, request, jsonify, render_template

from labelling.chess_analyzer import ChessGameAnalyzer, PlaystyleLabeler
from ffnn import ChessStyleFFNN

app = Flask(__name__)

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', '/usr/games/stockfish')
ANALYSIS_DEPTH = int(os.environ.get('ANALYSIS_DEPTH', 8))

RF_MODEL_PATH = 'models/RF/rf_baseline_model.pkl'
RF_SCALER_PATH = 'models/RF/scaler.pkl'
RF_FEATURES_PATH = 'models/RF/feature_columns.pkl'

FFNN_MODEL_PATH = 'models/FFNN/ffnn_model.pth'
FFNN_SCALER_PATH = 'models/FFNN/scaler.pkl'
FFNN_FEATURES_PATH = 'models/FFNN/feature_columns.pkl'
FFNN_LABEL_MAP = {0: 'aggressive', 1: 'positional', 2: 'defensive', 3: 'balanced'}

XGB_MODEL_PATH = 'models/XGB/xgb_model.pkl'
XGB_SCALER_PATH = 'models/XGB/scaler.pkl'
XGB_FEATURES_PATH = 'models/XGB/feature_columns.pkl'
XGB_ENCODER_PATH = 'models/XGB/label_encoder.pkl'


def load_rf():
    model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(RF_SCALER_PATH)
    feature_cols = joblib.load(RF_FEATURES_PATH)
    return model, scaler, feature_cols


def load_ffnn():
    scaler = joblib.load(FFNN_SCALER_PATH)
    feature_cols = joblib.load(FFNN_FEATURES_PATH)
    device = torch.device('cpu')
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


def load_xgb():
    model = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(XGB_SCALER_PATH)
    feature_cols = joblib.load(XGB_FEATURES_PATH)
    encoder = joblib.load(XGB_ENCODER_PATH)
    return model, scaler, feature_cols, encoder


def extract_features(pgn_string, color):
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)

    if game is None:
        raise ValueError("Could not parse PGN. Please check the format.")

    moves = list(game.mainline_moves())
    if len(moves) < 20:
        raise ValueError("Game is too short (minimum 20 moves required).")

    analyzer = ChessGameAnalyzer(STOCKFISH_PATH, ANALYSIS_DEPTH)
    features = analyzer.analyze_game(game, color)

    return features


def features_to_array(features, feature_cols):
    summary = PlaystyleLabeler.get_feature_summary(features)

    feature_map = {
        **summary,
        'checks_given': features.checks_given,
        'captures_made': features.captures_made,
        'material_sacrifices': features.material_sacrifices,
        'early_attacks': features.early_attacks,
        'prophylactic_moves': features.prophylactic_moves,
        'positional_sacrifices': features.positional_sacrifices,
        'simplifications': features.simplifications,
        'defensive_moves': features.defensive_moves,
        'counterattacks': features.counterattacks,
        'best_moves_found': features.best_moves_found,
        'tactical_shots': features.tactical_shots,
        'blunders': features.blunders,
        'retreat_moves': features.retreat_moves,
        'trades_when_losing': features.trades_when_losing,
        'passive_moves': features.passive_moves,
        'total_moves': features.total_moves,
    }

    return np.array([[feature_map.get(col, 0) for col in feature_cols]])


def predict_rf(X_raw, feature_cols):
    model, scaler, cols = load_rf()
    X = features_to_array(X_raw, cols)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]
    prob_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(model.classes_, probs)}
    return prediction, prob_dict


def predict_ffnn(X_raw, feature_cols):
    model, scaler, cols, device = load_ffnn()
    X = features_to_array(X_raw, cols)
    X_scaled = scaler.transform(X)
    tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted = int(torch.argmax(outputs, dim=1).item())
    prediction = FFNN_LABEL_MAP[predicted]
    prob_dict = {FFNN_LABEL_MAP[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)}
    return prediction, prob_dict


def predict_xgb(X_raw, feature_cols):
    model, scaler, cols, encoder = load_xgb()
    X = features_to_array(X_raw, cols)
    X_scaled = scaler.transform(X)
    pred_encoded = model.predict(X_scaled)[0]
    prediction = encoder.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(X_scaled)[0]
    classes = encoder.inverse_transform(range(len(probs)))
    prob_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}
    return prediction, prob_dict


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        pgn = data.get('pgn', '').strip()
        color_str = data.get('color', 'white').lower()

        if not pgn:
            return jsonify({'error': 'No PGN provided.'}), 400

        color = chess.WHITE if color_str == 'white' else chess.BLACK

        features = extract_features(pgn, color)

        if features.total_moves == 0:
            return jsonify({'error': 'No moves found for the selected colour.'}), 400

        rf_style, rf_probs   = predict_rf(features, None)
        ffnn_style, ffnn_probs = predict_ffnn(features, None)
        xgb_style, xgb_probs  = predict_xgb(features, None)

        return jsonify({
            'total_moves': features.total_moves,
            'results': {
                'Random Forest': {'style': rf_style,   'probabilities': rf_probs},
                'FFNN': {'style': ffnn_style, 'probabilities': ffnn_probs},
                'XGBoost': {'style': xgb_style,  'probabilities': xgb_probs},
            }
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError:
        return jsonify({'error': 'Stockfish engine not found. Check STOCKFISH_PATH.'}), 500
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)