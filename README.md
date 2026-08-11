# Chess Playing Style Classifier

A machine learning system that classifies a player's style in a single chess game as
**aggressive**, **positional**, **defensive**, or **balanced**, using Stockfish-derived
features extracted from PGN move data.

**Live demo:** https://chess-playing-style-classifier-production-ee70.up.railway.app/

Paste a PGN, pick a colour, and the app returns a predicted style and class probabilities
from all three trained models.

---

## How it works

```
PGN file ──► ChessGameAnalyzer ──► GameFeatures ──► PlaystyleLabeler ──► labelled CSV
             (Stockfish, per move)  (23 features)    (rule-based)             │
                                                                              ▼
                                                            ┌─────────────────────────┐
                                                            │ RF  ·  FFNN  ·  XGBoost │
                                                            └─────────────────────────┘
                                                                              │
                                                                              ▼
                                                                Flask app (/predict)
```

1. **Feature extraction** — every move by the chosen player is analysed by Stockfish
   before and after it is played. The centipawn delta plus board-state checks produce 23
   features (checks per move, captures per move, average centipawn loss, engine-match
   accuracy, sacrifices, early attacks, prophylactic moves, simplifications, tactical
   shots, blunders, retreats, trades when losing, passive moves, defensive moves,
   counterattacks, and raw counts).
2. **Labelling** — `PlaystyleLabeler` scores each style against tuned ratio thresholds
   (with rating-adjusted centipawn-loss cut-offs) and assigns the highest-scoring style.
   Anything with no clear winner is labelled `balanced`. This produces the ground-truth
   labels the supervised models are trained on.
3. **Classification** — three models are trained on the labelled dataset and served
   together so their predictions can be compared side by side.

---

## Results

Test-set accuracy on the held-out 20% split (4,000 samples):

| Model | Accuracy | Notes |
|---|---|---|
| **XGBoost** | **96.95%** | best across every class |
| Random Forest | 91.15% | 100 trees, balanced class weights |
| FFNN | 89.78% | 23 → 64 → 32 → 4, dropout 0.3, early stopping |

Per-class F1 (XGBoost): aggressive 0.98, positional 0.98, defensive 0.94, balanced 0.97.
`defensive` is consistently the hardest class for all three models — it is most often
confused with `aggressive`, which shares the capture-heavy signature.

Full metrics, confusion matrices and feature importances are in `results/` and
`models/*/confusion_matrix_*.png`.

### Cross-rating behaviour

`eval_prediction.py` runs all three models over two unseen rating bands (1,000 samples
each) to check that predictions behave sensibly outside the training distribution:

| Rating band | Aggressive | Positional | Defensive | Balanced | 3-model agreement |
|---|---|---|---|---|---|
| High (1601–2449) | 38.9% | 29.6% | 24.9% | 6.6% | 86.7% |
| Low (787–1399) | 55.1% | 15.4% | 24.2% | 5.3% | 85.9% |

(XGBoost distributions shown.) Lower-rated players are classified as aggressive far more
often and positional far less often, which matches the expected trend.

---

## Repository layout

| Path | Purpose |
|---|---|
| `labelling/chess_analyzer.py` | `ChessGameAnalyzer` (Stockfish feature extraction) and `PlaystyleLabeler` (rule-based labels) |
| `labelling/processing.py` | Batch-processes a PGN archive into a labelled CSV |
| `labelling/stats.py` | Prints summary statistics for a labelled CSV |
| `rf.py` | Trains the Random Forest baseline |
| `ffnn.py` | Defines and trains the feed-forward neural network |
| `nnprep.py` | Data preparation helpers for the FFNN |
| `xgb.py` | Trains the XGBoost classifier |
| `eval_prediction.py` | Cross-rating evaluation of all three models |
| `app.py` | Flask web app serving `/` and `/predict` |
| `templates/index.html` | Front end |
| `tests/test_*.py` | Per-model smoke tests that run a sample game end to end |
| `models/` | Trained model files, scalers, encoders, feature column lists |
| `results/` | Evaluation metrics (JSON), feature importances (CSV), charts |
| `report.pdf` | Full project report |

---

## Setup

Requires Python 3.11+ and a Stockfish binary on the system.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# macOS
brew install stockfish
# Debian/Ubuntu
sudo apt install stockfish
```

The app locates Stockfish via the `STOCKFISH_PATH` environment variable, falling back to
whatever `stockfish` resolves to on `PATH`, then `/usr/games/stockfish`.

### Run the web app

```bash
export STOCKFISH_PATH=$(which stockfish)
python app.py                     # http://localhost:5000
```

Production (as deployed):

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --preload --workers 2 --threads 2
```

### API

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"pgn": "[Event \"...\"] 1. e4 e5 ...", "color": "white"}'
```

```json
{
  "total_moves": 34,
  "results": {
    "Random Forest": { "style": "aggressive", "probabilities": { "aggressive": 61.0, "...": 0.0 } },
    "FFNN":          { "style": "aggressive", "probabilities": { "...": 0.0 } },
    "XGBoost":       { "style": "aggressive", "probabilities": { "...": 0.0 } }
  }
}
```

Games must be between 20 and 400 moves. Requests are rate limited and oversized bodies
are rejected before buffering.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `STOCKFISH_PATH` | auto-detected | Path to the Stockfish binary |
| `ANALYSIS_DEPTH` | `8` | Engine search depth per move (web app) |
| `MAX_PGN_BYTES` | `262144` | Maximum request body size |
| `MAX_GAME_MOVES` | `400` | Upper bound on moves analysed per request |
| `PREDICT_RATE_LIMIT` | `10 per minute; 100 per hour` | Rate limit on `/predict` |
| `RATELIMIT_STORAGE_URI` | `memory://` | Set to a shared backend (e.g. Redis) for a global limit across workers |
| `ENGINE_HASH_MB` | `128` | Stockfish hash table size |
| `ENGINE_THREADS` | `1` | Stockfish threads |

Defaults are deliberately small so a single request cannot monopolise the container.
Offline batch labelling passes much larger values (4 GB hash, 8 threads).

---

## Reproducing the pipeline

Training data comes from the [Lichess Open Database](https://database.lichess.org/).
Download a monthly PGN archive and decompress it first.

**1. Label a PGN archive** (run from inside `labelling/`, which is where its imports resolve):

```bash
cd labelling
python processing.py \
  --pgn /path/to/lichess_db_standard_rated_2017-01.pgn \
  --stockfish $(which stockfish) \
  --output labeled_dataset_2017.csv \
  --depth 12 --min-rating 1500 --max-games 5000
```

Each game yields two rows — one per player. Analysis is engine-bound, so expect this to
take hours for a few thousand games.

**2. Inspect the dataset:**

```bash
python labelling/stats.py --csv labelling/labeled_dataset_2017.csv
```

**3. Train the models** (each reads `labelling/labeled_dataset_2017.csv` and writes into
`models/` and `results/`):

```bash
python rf.py
python xgb.py
python ffnn.py
```

**4. Cross-rating evaluation** (uses the pre-built `tests/eval_high.csv` and
`tests/eval_low.csv`):

```bash
python eval_prediction.py
```

Note: the labelled CSVs are not committed (`labeled_data/` is gitignored), so step 1 must
be run before retraining. The trained artefacts in `models/` are committed, so the web app
works without retraining.

---

## Tests

Each test script runs a sample game through the full extract-and-predict path for one
model:

```bash
python tests/test_rf.py $(which stockfish)
python tests/test_xgb.py $(which stockfish)
python tests/test_ffnn.py $(which stockfish)

# or against your own game
python tests/test_rf.py $(which stockfish) --pgn my_game.pgn --color black
```

---

## Deployment

Deployed on Railway via Nixpacks. `nixpacks.toml` provisions Python 3.11 and Stockfish
from nixpkgs and starts gunicorn with `--preload`, so the three models load once before
forking and workers share them copy-on-write.
