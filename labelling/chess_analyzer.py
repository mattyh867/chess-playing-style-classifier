import chess
import chess.pgn
import chess.engine
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path


@dataclass
class GameFeatures:
    """Features extracted from a chess game"""
    # Aggression metrics
    checks_given: int = 0
    captures_made: int = 0
    material_sacrifices: int = 0
    early_attacks: int = 0  # attacks before move 15
    king_safety_risks: int = 0

    # Positional metrics
    prophylactic_moves: int = 0
    positional_sacrifices: int = 0

    # Tactical metrics
    avg_centipawn_loss: float = 0.0
    best_moves_found: int = 0
    tactical_shots: int = 0
    blunders: int = 0

    # Defensive metrics
    simplifications: int = 0
    defensive_moves: int = 0
    counterattacks: int = 0
    material_equality_maintained: int = 0
    retreat_moves: int = 0
    trades_when_losing: int = 0 
    passive_moves: int = 0 

    # General metrics
    total_moves: int = 0
    complex_positions: int = 0


class ChessGameAnalyzer:
    """Analyzes chess games using Stockfish"""
    
    def __init__(self, stockfish_path: str, depth: int = 18):
        """Initialize analyzer with Stockfish engine"""
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Hash": 4096})
        self.engine.configure({"Threads": 8}) 
        self.depth = depth
        
    def __del__(self):
        """Clean up engine"""
        if hasattr(self, 'engine'):
            self.engine.quit()
    
    def analyze_game(self, game: chess.pgn.Game, color: chess.Color) -> GameFeatures:
        """Analyze a game and extract features for one player"""
        features = GameFeatures()
        board = game.board()
        moves = list(game.mainline_moves())
        
        prev_eval = 0.0
        move_number = 0
        
        for move in moves:
            move_number += 1
            
            if board.turn != color:
                board.push(move)
                continue
            
            features.total_moves += 1
            
            try:
                info_before = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                eval_before = self._get_centipawn_score(info_before, board.turn)
                
                best_move = info_before.get("pv", [None])[0]
                
                board.push(move)
                
                info_after = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                eval_after = self._get_centipawn_score(info_after, not board.turn)
                
                cp_loss = eval_before - eval_after
                features.avg_centipawn_loss += abs(cp_loss)
                
                if move == best_move:
                    features.best_moves_found += 1
                
                self._analyze_move_type(board, move, features, move_number, eval_before, eval_after)
                
                if cp_loss < -200:
                    features.blunders += 1
                elif abs(cp_loss) < 20 and abs(eval_after) > 100:
                    features.tactical_shots += 1
                
                if self._is_complex_position(info_before):
                    features.complex_positions += 1
                
                prev_eval = eval_after
                
            except Exception as e:
                print(f"Error analyzing move {move_number}: {e}")
                board.push(move)
                continue
        
        if features.total_moves > 0:
            features.avg_centipawn_loss /= features.total_moves
        
        return features
    
    def _get_centipawn_score(self, info: Dict, turn: chess.Color) -> float:
        """Extract centipawn score from engine info"""
        score = info.get("score")
        if score is None:
            return 0.0
        
        # Handle mate scores
        if score.is_mate():
            mate_in = score.relative.mate()
            return 10000 if mate_in > 0 else -10000
        
        cp = score.relative.score()
        return cp if cp is not None else 0.0
    
    def _is_complex_position(self, info: Dict) -> bool:
        """Determine if position is complex based on engine analysis"""
        # Check if there are multiple moves with similar evaluations
        pv = info.get("pv", [])
        return len(pv) > 0 and len(pv) < 3 
    
    def _analyze_move_type(self, board: chess.Board, move: chess.Move, features: GameFeatures, move_number: int,
                           eval_before: float, eval_after: float):
        """Analyze what type of move was played"""
        
        board.pop()
        
        board.push(move)
        if board.is_check():
            features.checks_given += 1
        board.pop()
        
        if board.is_capture(move):
            features.captures_made += 1
            
            piece_value = self._get_piece_value(board.piece_at(move.from_square))
            captured_value = self._get_piece_value(board.piece_at(move.to_square))
            
            if piece_value > captured_value and eval_after > eval_before - 50:
                features.material_sacrifices += 1
        
        if move_number <= 15:
            if self._is_attacking_move(board, move):
                features.early_attacks += 1
        
        if board.piece_at(move.from_square).piece_type == chess.KING:
            if move_number < 10:
                features.king_safety_risks += 1
        
        if board.is_capture(move) and eval_before >= -50:
            features.simplifications += 1
        
        if not board.is_capture(move) and not board.is_check() and eval_after > eval_before:
            if eval_after - eval_before < 50:
                features.prophylactic_moves += 1

        if board.piece_at(move.from_square).piece_type == chess.PAWN:
            if board.is_capture(move) and eval_after > eval_before:
                features.positional_sacrifices += 1
        
        # Defensive moves
        if eval_before < -100:
            if not board.is_capture(move) and not self._is_attacking_move(board, move):
                features.defensive_moves += 1
            elif board.is_capture(move):
                features.counterattacks += 1
        
        if self._is_retreat_move(board, move, eval_before):
            features.retreat_moves += 1
        
        if self._is_trades_when_losing(board, move, eval_before):
            features.trades_when_losing += 1
        
        if self._is_passive_move(board, move, eval_before, eval_after):
            features.passive_moves += 1
        
        board.push(move)
    
    def _get_piece_value(self, piece) -> int:
        if piece is None:
            return 0
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        return values.get(piece.piece_type, 0)
    
    def _is_attacking_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move is attacking"""
        board.push(move)
        
        attacks = False
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    attacks = True
                    break
        
        board.pop()
        return attacks
    
    def _is_retreat_move(self, board: chess.Board, move: chess.Move, eval_before: float) -> bool:
        """Check if move is a retreat"""
        if eval_before > 100:
            return False
        
        piece = board.piece_at(move.from_square)
        if piece is None:
            return False
        
        was_attacked = board.is_attacked_by(not board.turn, move.from_square)
        
        if not was_attacked:
            return False
        
        board.push(move)
        
        is_attacked_after = board.is_attacked_by(board.turn, move.to_square)
        
        board.pop()
        
        return was_attacked and not is_attacked_after
    
    def _is_trades_when_losing(self, board: chess.Board, move: chess.Move, 
                               eval_before: float) -> bool:
        """Check if trading pieces when in a losing position"""
        if eval_before > -100:
            return False
        
        if not board.is_capture(move):
            return False
        
        return True
    
    def _is_passive_move(self, board: chess.Board, move: chess.Move, 
                        eval_before: float, eval_after: float) -> bool:
        """Check if move is passive/waiting"""
        if board.is_capture(move):
            return False
        
        board.push(move)
        is_check = board.is_check()
        board.pop()
        
        if is_check:
            return False
        
        # Small or no evaluation change
        eval_change = abs(eval_after - eval_before)
        if eval_change > 20:
            return False
        
        # Not creating immediate threats
        if self._is_attacking_move(board, move):
            return False
        
        return True


class PlaystyleLabeler:
    # Thresholds for classification
    AGGRESSIVE_THRESHOLDS = {
        'checks_ratio': 0.14,
        'captures_ratio': 0.24,      
        'sacrifices_ratio': 0.04,  
        'early_attacks': 2      
    }
    
    POSITIONAL_THRESHOLDS = {
        'prophylactic_ratio': 0.08,  
        'positional_sacrifices_ratio': 0.012, 
        'best_moves_ratio': 0.28  
    }
    
    DEFENSIVE_THRESHOLDS = {
        'simplifications_ratio': 0.16, 
        'defensive_ratio': 0.10, 
        'counterattacks': 1, 
        'retreat_ratio': 0.09, 
        'trades_when_losing_ratio': 0.12,
        'passive_ratio': 0.13, 
    }

    @staticmethod
    def _get_dynamic_cp_threshold(elo: int, base_threshold: float) -> float:
        # Adjust CP loss threshold based on player rating
        # Higher rating = stricter threshold
        if elo >= 2400:
            return base_threshold * 0.7
        elif elo >= 2200:
            return base_threshold * 0.85
        elif elo >= 2000:
            return base_threshold * 1.0
        else:
            return base_threshold * 1.2
    
    @staticmethod
    def label_game(features: GameFeatures, player_elo: int = None) -> str:
        if features.total_moves == 0:
            return 'unknown'
        
        scores = {
            'aggressive': 0,
            'positional': 0,
            'defensive': 0
        }
        
        # Calculate ratios
        checks_ratio = features.checks_given / features.total_moves
        captures_ratio = features.captures_made / features.total_moves
        sacrifices_ratio = features.material_sacrifices / features.total_moves
        prophylactic_ratio = features.prophylactic_moves / features.total_moves
        positional_sac_ratio = features.positional_sacrifices / features.total_moves
        best_moves_ratio = features.best_moves_found / features.total_moves
        simplifications_ratio = features.simplifications / features.total_moves
        defensive_ratio = features.defensive_moves / features.total_moves
        retreat_ratio = features.retreat_moves / features.total_moves
        trades_when_losing_ratio = features.trades_when_losing / features.total_moves
        passive_ratio = features.passive_moves / features.total_moves
        
        # Adjust thresholds if ELO provided
        if player_elo:
            aggressive_cp_max = PlaystyleLabeler._get_dynamic_cp_threshold(
                player_elo, 45
            )
            positional_cp_max = PlaystyleLabeler._get_dynamic_cp_threshold(
                player_elo, 35
            )
        else:
            aggressive_cp_max = 45
            positional_cp_max = 35

        # Score aggressive style
        if checks_ratio >= PlaystyleLabeler.AGGRESSIVE_THRESHOLDS['checks_ratio']:
            scores['aggressive'] += 1
        if captures_ratio >= PlaystyleLabeler.AGGRESSIVE_THRESHOLDS['captures_ratio']:
            scores['aggressive'] += 1
        if sacrifices_ratio >= PlaystyleLabeler.AGGRESSIVE_THRESHOLDS['sacrifices_ratio']:
            scores['aggressive'] += 1
        if features.early_attacks >= PlaystyleLabeler.AGGRESSIVE_THRESHOLDS['early_attacks']:
            scores['aggressive'] += 1
        if features.avg_centipawn_loss <= aggressive_cp_max:
            scores['aggressive'] += 0.5
        
        # Score positional style
        if prophylactic_ratio >= PlaystyleLabeler.POSITIONAL_THRESHOLDS['prophylactic_ratio']:
            scores['positional'] += 1.5
        if positional_sac_ratio >= PlaystyleLabeler.POSITIONAL_THRESHOLDS['positional_sacrifices_ratio']:
            scores['positional'] += 1
        if features.avg_centipawn_loss <= positional_cp_max:
            scores['positional'] += 1
        if best_moves_ratio >= PlaystyleLabeler.POSITIONAL_THRESHOLDS['best_moves_ratio']:
            scores['positional'] += 1
        
        # Score defensive style
        if simplifications_ratio >= PlaystyleLabeler.DEFENSIVE_THRESHOLDS['simplifications_ratio']:
            scores['defensive'] += 1
        if defensive_ratio >= PlaystyleLabeler.DEFENSIVE_THRESHOLDS['defensive_ratio']:
            scores['defensive'] += 1
        if features.counterattacks >= PlaystyleLabeler.DEFENSIVE_THRESHOLDS['counterattacks']:
            scores['defensive'] += 1
        if retreat_ratio >= PlaystyleLabeler.DEFENSIVE_THRESHOLDS['retreat_ratio']:
            scores['defensive'] += 1
        if trades_when_losing_ratio >= PlaystyleLabeler.DEFENSIVE_THRESHOLDS['trades_when_losing_ratio']:
            scores['defensive'] += 1
        if passive_ratio >= PlaystyleLabeler.DEFENSIVE_THRESHOLDS['passive_ratio']:
            scores['defensive'] += 1
        
        # Determine label
        max_score = max(scores.values())
        
        # If no clear winner or low scores, it's balanced
        if max_score < 2.0:
            return 'balanced'
        
        # Return the style with highest score
        for style, score in scores.items():
            if score == max_score:
                return style
        
        return 'balanced'
    
    @staticmethod
    def get_feature_summary(features: GameFeatures) -> Dict:
        if features.total_moves == 0:
            return {}
        
        return {
            'checks_per_move': features.checks_given / features.total_moves,
            'captures_per_move': features.captures_made / features.total_moves,
            'avg_centipawn_loss': features.avg_centipawn_loss,
            'accuracy': features.best_moves_found / features.total_moves,
            'sacrifices': features.material_sacrifices,
            'early_attacks': features.early_attacks,
            'prophylactic_moves': features.prophylactic_moves,
            'simplifications': features.simplifications,
            'tactical_shots': features.tactical_shots,
            'blunders': features.blunders,
            'retreat_moves': features.retreat_moves,
            'trades_when_losing': features.trades_when_losing,
            'passive_moves': features.passive_moves,
            'defensive_moves': features.defensive_moves,
            'counterattacks': features.counterattacks,
            'positional_sacrifices': features.positional_sacrifices
        }