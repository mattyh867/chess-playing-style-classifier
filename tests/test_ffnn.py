# Usage: python tests/test_ffnn.py /path/to/stockfish
#        python tests/test_ffnn.py /path/to/stockfish --pgn path/to/game.pgn
#        python tests/test_ffnn.py /path/to/stockfish --pgn path/to/game.pgn --color white

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import chess
import chess.pgn
from io import StringIO
from labelling.chess_analyzer import ChessGameAnalyzer, PlaystyleLabeler
import numpy as np
import joblib
import torch
import argparse

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

SAMPLE_GAME = """
[Event "Rated Bullet game"]
[Site "https://lichess.org/ulpcm79x"]
[White "journal"]
[Black "Kiriush33"]
[Result "1-0"]
[UTCDate "2013.12.31"]
[UTCTime "23:00:20"]
[WhiteElo "1691"]
[BlackElo "1717"]
[WhiteRatingDiff "+13"]
[BlackRatingDiff "-12"]
[ECO "C20"]
[Opening "King's Pawn Game: Leonardis Variation"]
[TimeControl "60+0"]
[Termination "Time forfeit"]

1. e4 e5 2. d3 Nf6 3. Nf3 d5 4. Nc3 Nc6 5. exd5 Nxd5 6. Nxd5 Qxd5 7. c4 Bb4+ 8. Bd2 Bxd2+ 9. Qxd2 Qd7 10. Be2 Qd6 11. O-O Bf5 12. Rfe1 O-O-O 13. b3 f6 14. Qe3 Bxd3 15. Bxd3 Qxd3 16. Qxd3 Rxd3 17. Nh4 Rhd8 18. Nf5 Rd8d7 19. Kh1 Nb4 20. Rec1 Rd3d2 21. a3 Nd3 22. Rcd1 Nxf2+ 23. Kg1 Nxd1 24. h3 Nc3 25. Rf1 Ne4 26. b4 Rd2d1 27. Ne7+ Rxe7 28. Rxd1 Rd7 29. Rxd7 1-0
"""


def analyze_game(stockfish_path, game, color):
    analyzer = ChessGameAnalyzer(stockfish_path, depth=15)
    features = analyzer.analyze_game(game, color)
    summary = PlaystyleLabeler.get_feature_summary(features)
    
    feature_data = {
        **summary,
        'checks_given': features.checks_given,
        'captures_made': features.captures_made,
        'material_sacrifices': features.material_sacrifices,
        'early_attacks': features.early_attacks,
        'king_safety_risks': features.king_safety_risks,
        'prophylactic_moves': features.prophylactic_moves,
        'positional_sacrifices': features.positional_sacrifices,
        'simplifications': features.simplifications,
        'defensive_moves': features.defensive_moves,
        'counterattacks': features.counterattacks,
        'best_moves_found': features.best_moves_found,
        'tactical_shots': features.tactical_shots,
        'blunders': features.blunders,
        'complex_positions': features.complex_positions,
        'total_moves': features.total_moves
    }
    
    rule_label = PlaystyleLabeler.label_game(features)
    return feature_data, rule_label


def features_to_model_input(feature_data, feature_cols):
    values = []
    for col in feature_cols:
        if col in feature_data:
            values.append(feature_data[col])
        else:
            print(f"  WARNING: Feature '{col}' not found, using 0")
            values.append(0)
    return np.array(values).reshape(1, -1)


def main():
    parser = argparse.ArgumentParser(description='Test FFNN on a single chess game')
    parser.add_argument('stockfish', help='Path to Stockfish binary')
    parser.add_argument('--pgn', default=None, help='Path to PGN file')
    parser.add_argument('--color', default='both', choices=['white', 'black', 'both'],
                       help='Which color to analyze (default: both)')
    args = parser.parse_args()
    
    print("\nFEEDFORWARD NEURAL NETWORK - SINGLE GAME TEST")
    
    if args.pgn:
        print(f"\nLoading game from: {args.pgn}")
        with open(args.pgn) as f:
            game = chess.pgn.read_game(f)
    else:
        print("\nUsing sample game (no --pgn provided)")
        game = chess.pgn.read_game(StringIO(SAMPLE_GAME))
    
    if game is None:
        print("ERROR: Could not parse game")
        sys.exit(1)
    
    print(f"\nGame Info:")
    print(f"  White: {game.headers.get('White', '?')} ({game.headers.get('WhiteElo', '?')})")
    print(f"  Black: {game.headers.get('Black', '?')} ({game.headers.get('BlackElo', '?')})")
    print(f"  Opening: {game.headers.get('Opening', '?')}")
    print(f"  Result: {game.headers.get('Result', '?')}")
    print(f"  Moves: {len(list(game.mainline_moves()))}")
    
    print(f"\nLoading FFNN model...")
    feature_cols = joblib.load(os.path.join(BASE_DIR, 'models/FFNN/feature_columns.pkl'))
    
    from train_ffnn import ChessStyleFFNN
    
    metadata_path = os.path.join(BASE_DIR, 'models/FFNN/ffnn_metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = joblib.load(f)

    
    arch = metadata['architecture']
    model = ChessStyleFFNN(
        input_size=len(feature_cols),
        hidden1_size=arch['hidden1_size'],
        hidden2_size=arch['hidden2_size'],
        num_classes=4,
        dropout_rate=arch['dropout_rate']
    )
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models/FFNN/ffnn_model.pth'), map_location='cpu'))
    model.eval()
    
    scaler_path = os.path.join(BASE_DIR, 'models/FFNN/scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)
    
    print(f"  Model loaded ({len(feature_cols)} features)")
    
    label_mapping = {
        0: 'aggressive',
        1: 'positional',
        2: 'defensive',
        3: 'balanced'
    }
    
    colors_to_analyze = []
    if args.color in ['white', 'both']:
        colors_to_analyze.append(('White', chess.WHITE))
    if args.color in ['black', 'both']:
        colors_to_analyze.append(('Black', chess.BLACK))
    
    for color_name, color in colors_to_analyze:
        print(f"\nAnalyzing {color_name}...")
        print(f"  Player: {game.headers.get(color_name, '?')}")
        print(f"  Rating: {game.headers.get(f'{color_name}Elo', '?')}")
        
        feature_data, rule_label = analyze_game(args.stockfish, game, color)
        
        print(f"\nKey Metrics:")
        for key, label in [('captures_per_move', 'Captures/move'), ('checks_per_move', 'Checks/move'),
                           ('avg_centipawn_loss', 'Avg CP loss'), ('accuracy', 'Accuracy'),
                           ('prophylactic_moves', 'Prophylactic'), ('defensive_moves', 'Defensive'),
                           ('blunders', 'Blunders'), ('total_moves', 'Total moves')]:
            if key in feature_data:
                val = feature_data[key]
                print(f"    {label:<22} {val:.3f}" if isinstance(val, float) else f"    {label:<22} {val}")
        
        X = features_to_model_input(feature_data, feature_cols)
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        prediction = label_mapping[predicted_class]
        
        print(f"\nPrediction: {prediction.upper()}")
        print(f"  Confidence: {max(probabilities)*100:.1f}%")
        print(f"  Rule-based label: {rule_label.upper()}")
        
        print(f"\nClass Probabilities:")
        style_probs = [(label_mapping[i], probabilities[i]) for i in range(4)]
        for cls, prob in sorted(style_probs, key=lambda x: -x[1]):
            print(f"    {cls:<14} {prob*100:>6.1f}%")
    
    print(f"\nTEST COMPLETE")


if __name__ == "__main__":
    main()