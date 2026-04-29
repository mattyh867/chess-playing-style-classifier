# Chess Playing Style Detector

A machine learning system that classifies chess playing styles 
(aggressive, positional, defensive, balanced) from game data.

Deployed at:
https://chess-playing-style-classifier-production-ee70.up.railway.app/

## Scripts

- `processing.py` / `chess_analyzer.py` — data pipeline and Stockfish feature extraction
- `rf.py` — Random Forest classifier
- `ffnn.py` / `nnprep.py` — Feed-Forward Neural Network
- `xgb.py` — Extreme Gradient Boosting classifier
- `eval_prediction.py` — cross-skill evaluation

## Setup

```bash
pip install -r requirements.txt
```

Stockfish must be installed separately and the path configured in `chess_analyzer.py`.

## Data

Training data sourced from the Lichess Open Database (2015). 
Dataset CSV files are not included in this repository.
