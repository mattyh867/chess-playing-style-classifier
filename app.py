import os
import io
import joblib
import numpy as np
import chess
import chess.pgn
from flask import Flask, request, jsonify, render_template

from chess_analyzer import ChessGameAnalyzer, PlaystyleLabeler

app = Flask(__name__)

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', '/usr/games/stockfish')
ANALYSIS_DEPTH = int(os.environ.get('ANALYSIS_DEPTH', 12))

RF_MODEL_PATH = 'models/RF/rf_baseline_model.pkl'
RF_SCALER_PATH = 'models/RF/scaler.pkl'
RF_FEATURES_PATH = 'models/RF/feature_columns.pkl'


def load_model():
    model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(RF_SCALER_PATH)
    feature_cols = joblib.load(RF_FEATURES_PATH)
    return model, scaler, feature_cols


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

        model, scaler, feature_cols = load_model()
        X = features_to_array(features, feature_cols)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        classes = model.classes_

        prob_dict = {cls: round(float(prob) * 100, 1)
                     for cls, prob in zip(classes, probabilities)}

        return jsonify({
            'style': prediction,
            'probabilities': prob_dict,
            'total_moves': features.total_moves,
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError:
        return jsonify({'error': 'Stockfish engine not found. Check STOCKFISH_PATH.'}), 500
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)