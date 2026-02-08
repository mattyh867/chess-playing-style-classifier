import chess
import chess.pgn
from chess_analyzer import ChessGameAnalyzer, PlaystyleLabeler, GameFeatures
import pandas as pd
from typing import List, Dict


class BatchGameProcessor:
    
    def __init__(self, stockfish_path: str, depth: int = 18, min_rating: int = 2000, max_games: int = 5000):
        self.analyzer = ChessGameAnalyzer(stockfish_path, depth)
        self.labeler = PlaystyleLabeler()
        self.min_rating = min_rating
        self.max_games = max_games
        
    def process_pgn_file(self, pgn_path: str, output_path: str = None, save_interval: int = 100) -> pd.DataFrame:
        results = []
        games_processed = 0
        games_read = 0 
        skipped_rating = 0
        skipped_short = 0
        
        print(f"Processing PGN file: {pgn_path}")
        print(f"Min rating: {self.min_rating}, Max games: {self.max_games}")
        
        with open(pgn_path) as pgn_file:
            while games_processed < self.max_games:
                game = chess.pgn.read_game(pgn_file)
                
                if game is None:
                    break

                games_read += 1
                
                # Filter by rating
                headers = game.headers
                try:
                    # Handle '?' ratings from PGN files
                    white_elo_str = headers.get("WhiteElo", "0")
                    black_elo_str = headers.get("BlackElo", "0")

                    white_elo = int(white_elo_str) if white_elo_str != '?' else 0
                    black_elo = int(black_elo_str) if black_elo_str != '?' else 0

                    if white_elo < self.min_rating or black_elo < self.min_rating:
                        skipped_rating += 1
                        continue
                    
                    # Skip if game was too short or ended in draw/resignation too early
                    result = headers.get("Result", "*")
                    if len(list(game.mainline_moves())) < 20:
                        continue
                    
                    white_data = self._process_player(game, chess.WHITE, headers)
                    black_data = self._process_player(game, chess.BLACK, headers)
                    
                    if white_data:
                        results.append(white_data)
                    if black_data:
                        results.append(black_data)
                    
                    games_processed += 1

                    if games_read % 100 == 0:  # Log every 100 games
                            print(f"Read {games_read} games, skipped {skipped_rating} due to rating")
                    
                    if games_processed % 10 == 0:
                        print(f"Processed {games_processed} games... "
                              f"({len(results)} player datasets created)")
                    
                    if output_path and games_processed % save_interval == 0:
                        self._save_results(results, output_path, intermediate=True)
                    
                except Exception as e:
                    print(f"Error processing game: {e}")
                    continue
        
        df = pd.DataFrame(results)
        
        if output_path:
            self._save_results(results, output_path, intermediate=False)
        
        print(f"\nProcessing complete!")
        print(f"Total games processed: {games_processed}")
        print(f"Total player datasets: {len(results)}")
        self._print_statistics(df)
        
        return df
    
    def _process_player(self, game: chess.pgn.Game, color: chess.Color, headers: Dict) -> Dict:
        try:
            features = self.analyzer.analyze_game(game, color)
            
            label = self.labeler.label_game(features)
            
            summary = self.labeler.get_feature_summary(features)
            
            player_name = headers.get("White" if color == chess.WHITE else "Black", "Unknown")
            player_elo_str = headers.get("WhiteElo" if color == chess.WHITE else "BlackElo", "0")
            player_elo = int(player_elo_str) if player_elo_str != '?' else 0
            
            data = {
                'player_name': player_name,
                'player_elo': player_elo,
                'color': 'white' if color == chess.WHITE else 'black',
                'game_id': headers.get("Site", ""),
                'date': headers.get("Date", ""),
                'result': headers.get("Result", "*"),
                'opening': headers.get("Opening", "Unknown"),
                'label': label,
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
            
            return data
            
        except Exception as e:
            print(f"Error analyzing player: {e}")
            return None
    
    def _save_results(self, results: List[Dict], output_path: str, 
                     intermediate: bool = False):
        df = pd.DataFrame(results)
        save_path = output_path
        
        df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")
    
    def _print_statistics(self, df: pd.DataFrame):
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        if 'label' in df.columns:
            print("\nPlaystyle Distribution:")
            print(df['label'].value_counts())
            print(f"\nPercentages:")
            print(df['label'].value_counts(normalize=True) * 100)
        
        if 'player_elo' in df.columns:
            print(f"\nRating Statistics:")
            print(f"Mean: {df['player_elo'].mean():.0f}")
            print(f"Median: {df['player_elo'].median():.0f}")
            print(f"Std: {df['player_elo'].std():.0f}")
        
        if 'avg_centipawn_loss' in df.columns:
            print(f"\nAccuracy Statistics:")
            print(f"Mean CP Loss: {df['avg_centipawn_loss'].mean():.1f}")
            print(f"Mean Accuracy: {df['accuracy'].mean()*100:.1f}%")
        
        print("="*50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze chess games and label by playstyle')
    parser.add_argument('--pgn', required=True, help='Path to PGN file')
    parser.add_argument('--stockfish', required=True, help='Path to Stockfish binary')
    parser.add_argument('--output', default='labeled_games.csv', help='Output CSV path')
    parser.add_argument('--depth', type=int, default=12, help='Analysis depth')
    parser.add_argument('--min-rating', type=int, default=1500, help='Minimum player rating')
    parser.add_argument('--max-games', type=int, default=5000, help='Maximum games to process')
    
    args = parser.parse_args()
    
    processor = BatchGameProcessor(
        stockfish_path=args.stockfish,
        depth=args.depth,
        min_rating=args.min_rating,
        max_games=args.max_games
    )
    
    df = processor.process_pgn_file(args.pgn, args.output)
    
    print(f"\nDataset saved to {args.output}")
    print(f"Total entries: {len(df)}")


if __name__ == "__main__":
    main()