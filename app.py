import os
import io
import uuid
import logging
import joblib
import numpy as np
import chess
import chess.pgn
import torch
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
import shutil

from labelling.chess_analyzer import ChessGameAnalyzer, PlaystyleLabeler
from ffnn import ChessStyleFFNN

app = Flask(__name__)

logger = logging.getLogger(__name__)

# Reject oversized bodies before they are buffered into memory.
MAX_PGN_BYTES = int(os.environ.get('MAX_PGN_BYTES', 256 * 1024))
app.config['MAX_CONTENT_LENGTH'] = MAX_PGN_BYTES

# Upper bound on analysis work per request. Each analysed move costs two
# Stockfish evaluations, so an unbounded game is a denial-of-service vector.
MAX_GAME_MOVES = int(os.environ.get('MAX_GAME_MOVES', 400))
MIN_GAME_MOVES = 20

# In-memory storage is per-process, so with N gunicorn workers the effective
# limit is N x this value. Set RATELIMIT_STORAGE_URI to a shared backend
# (e.g. Redis) for a true global limit.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=os.environ.get('RATELIMIT_STORAGE_URI', 'memory://'),
)
PREDICT_RATE_LIMIT = os.environ.get('PREDICT_RATE_LIMIT', '10 per minute; 100 per hour')

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH') or shutil.which('stockfish') or '/usr/games/stockfish'
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
    # weights_only=True prevents arbitrary code execution during unpickling.
    model.load_state_dict(
        torch.load(FFNN_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    return model, scaler, feature_cols, device


def load_xgb():
    model = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(XGB_SCALER_PATH)
    feature_cols = joblib.load(XGB_FEATURES_PATH)
    encoder = joblib.load(XGB_ENCODER_PATH)
    return model, scaler, feature_cols, encoder


# Loaded once at import so each request does not re-read every artefact from
# disk. With gunicorn --preload this happens before forking, so workers share
# the memory copy-on-write.
RF_BUNDLE = load_rf()
FFNN_BUNDLE = load_ffnn()
XGB_BUNDLE = load_xgb()


def extract_features(pgn_string, color):
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)

    if game is None:
        raise ValueError("Could not parse PGN. Please check the format.")

    moves = list(game.mainline_moves())
    if len(moves) < MIN_GAME_MOVES:
        raise ValueError(
            f"Game is too short (minimum {MIN_GAME_MOVES} moves required)."
        )
    if len(moves) > MAX_GAME_MOVES:
        raise ValueError(
            f"Game is too long (maximum {MAX_GAME_MOVES} moves supported)."
        )

    # try/finally so the Stockfish subprocess is always reaped, even if
    # analysis raises partway through.
    analyzer = ChessGameAnalyzer(STOCKFISH_PATH, ANALYSIS_DEPTH)
    try:
        return analyzer.analyze_game(game, color)
    finally:
        analyzer.close()


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
    model, scaler, cols = RF_BUNDLE
    X = features_to_array(X_raw, cols)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]
    prob_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(model.classes_, probs)}
    return prediction, prob_dict


def predict_ffnn(X_raw, feature_cols):
    model, scaler, cols, device = FFNN_BUNDLE
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
    model, scaler, cols, encoder = XGB_BUNDLE
    X = features_to_array(X_raw, cols)
    X_scaled = scaler.transform(X)
    pred_encoded = model.predict(X_scaled)[0]
    prediction = encoder.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(X_scaled)[0]
    classes = encoder.inverse_transform(range(len(probs)))
    prob_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}
    return prediction, prob_dict


@app.after_request
def set_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    response.headers.setdefault(
        'Content-Security-Policy',
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    )
    return response


@app.errorhandler(413)
def payload_too_large(e):
    return jsonify({'error': 'PGN is too large.'}), 413


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'error': 'Too many requests. Please wait and try again.'}), 429


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@limiter.limit(PREDICT_RATE_LIMIT)
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object.'}), 400

        pgn = data.get('pgn', '')
        color_str = data.get('color', 'white')
        if not isinstance(pgn, str) or not isinstance(color_str, str):
            return jsonify({'error': "'pgn' and 'color' must be strings."}), 400

        pgn = pgn.strip()
        color_str = color_str.lower()

        if not pgn:
            return jsonify({'error': 'No PGN provided.'}), 400

        if len(pgn.encode('utf-8')) > MAX_PGN_BYTES:
            return jsonify({'error': 'PGN is too large.'}), 400

        if color_str not in ('white', 'black'):
            return jsonify({'error': "'color' must be 'white' or 'black'."}), 400

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

    except HTTPException:
        # 413 (oversized body) and 429 (rate limited) are raised inside this
        # block; let Flask's error handlers render them instead of swallowing
        # them into a generic 500.
        raise
    except ValueError as e:
        # Deliberate, user-facing validation messages only.
        return jsonify({'error': str(e)}), 400
    except Exception:
        # Log the detail server-side; return only an opaque reference so
        # internals (paths, library versions) are not disclosed.
        error_id = uuid.uuid4().hex[:12]
        logger.exception('Analysis failed (error_id=%s)', error_id)
        return jsonify({
            'error': 'Analysis failed due to an internal error.',
            'error_id': error_id,
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)