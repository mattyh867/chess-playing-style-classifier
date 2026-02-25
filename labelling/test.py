import chess.pgn
from io import StringIO
from chess_analyzer import ChessGameAnalyzer, PlaystyleLabeler

# Sample PGN game
SAMPLE_GAME = """
[Event "Rated Blitz game"]
[Site "https://lichess.org/abc123"]
[Date "2024.01.15"]
[White "AggressivePlayer"]
[Black "PositionalPlayer"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2180"]
[Opening "Philidor Defense"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O
6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7
11. O-O Nxc3 12. Rxc3 e5 13. dxe5 Nxe5 14. Nxe5 Qxe5
15. Qc2 Qe7 16. Rfc1 Be6 17. Bd3 Rfd8 18. h3 Rac8
19. Rc5 Bd5 20. R1c3 Qd6 21. Bf5 Rc7 22. Rg3 g6
23. Bd3 Rdc8 24. Rf3 Kg7 25. Qc1 Qe7 26. Rfc3 Qd6
27. Rxd5 cxd5 28. Rxc7 Rxc7 29. Qxc7 Qxc7 30. 1-0
"""


def test_analyzer(stockfish_path: str):
    """Test the analyzer with a sample game"""
    print("="*50)
    print("TESTING CHESS ANALYZER PIPELINE")
    print("="*50)
    
    # Parse sample game
    pgn = StringIO(SAMPLE_GAME)
    game = chess.pgn.read_game(pgn)
    
    print("\nGame Info:")
    print(f"White: {game.headers['White']} ({game.headers['WhiteElo']})")
    print(f"Black: {game.headers['Black']} ({game.headers['BlackElo']})")
    print(f"Opening: {game.headers['Opening']}")
    print(f"Result: {game.headers['Result']}")
    
    # Initialize analyzer
    print("\nInitializing Stockfish...")
    try:
        analyzer = ChessGameAnalyzer(stockfish_path, depth=15)
        print("Stockfish loaded successfully")
    except Exception as e:
        print(f"Error loading Stockfish: {e}")
        print("\nMake sure to provide correct Stockfish path:")
        print("python test_pipeline.py /path/to/stockfish")
        return
    
    # Analyze White's play
    print("\n" + "-"*50)
    print("Analyzing White's playstyle...")
    print("-"*50)
    
    try:
        white_features = analyzer.analyze_game(game, chess.WHITE)
        white_label = PlaystyleLabeler.label_game(white_features)
        white_summary = PlaystyleLabeler.get_feature_summary(white_features)
        
        print(f"\nWhite's Playstyle: {white_label.upper()}")
        print("\nKey Metrics:")
        for key, value in white_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error analyzing White: {e}")
        return
    
    # Analyze Black's play
    print("\n" + "-"*50)
    print("Analyzing Black's playstyle...")
    print("-"*50)
    
    try:
        black_features = analyzer.analyze_game(game, chess.BLACK)
        black_label = PlaystyleLabeler.label_game(black_features)
        black_summary = PlaystyleLabeler.get_feature_summary(black_features)
        
        print(f"\nBlack's Playstyle: {black_label.upper()}")
        print("\nKey Metrics:")
        for key, value in black_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error analyzing Black: {e}")
        return
    
    print("\n" + "="*50)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py /path/to/stockfish")
        sys.exit(1)
    
    stockfish_path = sys.argv[1]
    test_analyzer(stockfish_path)