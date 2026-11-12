# self-trained-chessbot

A small decoder-only transformer (~10–25M parameters) trained **from scratch** on Lichess game records. It is never told the rules of chess: it only learns to predict the next character of a game's movetext (`1-0 1.e4 e5 2.Nf3 Nc6 …`). [python-chess](https://python-chess.readthedocs.io/) acts as the referee.

## Final product

1. **The trained model + a results table:** legal-move rate (overall and by move number), Lichess puzzle accuracy by rating, score vs Stockfish levels 1–3, one controlled experiment, and before/after fine-tuning.
2. **A self-hosted website to play against it:** the model runs in the browser (ONNX Runtime Web), chess.js checks legality, and the page shows the model's top-5 candidate moves with probabilities, its illegal attempts, a temperature slider and a base/fine-tuned toggle.

## How it works

- **Data.** Rated blitz, rapid and classical games where both players are rated 1800+, streamed from the [Lichess database](https://database.lichess.org) (2026-10 dump) and rewritten as one line per game: `1-0 1.e4 e5 2.Nf3 …`. The result comes first, so at play time the model can be asked to play like the side that won.
- **Tokens.** One token per character: 33 symbols (SAN letters, digits, `+#=-/.`, space, and a newline between games), stored as `uint8` arrays nanoGPT-style.
- **Model.** GPT-2-style decoder with a 1,024-character context, in two sizes: **base** (6 layers, 6 heads, 384-dim, 10.7M parameters) and **large** (8 layers, 8 heads, 512-dim, 25.2M parameters).
- **Playing.** To choose a move, the model writes characters until a space. The text counts as a move only if it is **exactly** the SAN of a legal move; otherwise the attempt is recorded and it tries again. Separately, every legal move is scored by the model's probability of writing it, which gives the top-5 list.
- **Training.** On Kaggle (one T4 GPU, fp16 mixed precision, ~100k tokens/s), driven from a laptop with the Kaggle CLI.

## Status

| Week | Focus | Status |
|------|-------|--------|
| W0 | Setup | done |
| W1 | Data pipeline + tokenizer | done: 3M games, 1.26B tokens |
| W2 | Model, training loop, Kaggle pipeline | done |
| W3 | Training discipline + experiment | done: base and large runs (model-size experiment) |
| W4 | Play + eval harness | done: legal rate, puzzles, Stockfish games |
| W5 | Fine-tune + write-up | fine-tunes done; write-up in progress |
| W6 | Web demo | working locally; new models, temperature slider and hosting to do |

## Results

Four models: **base** and **large** trained from scratch on the same 3M games, and each one fine-tuned on long games (**ft-base**, **ft-large**). All numbers are measured on data the models never trained on.

| | base | large | ft-base | ft-large |
|---|---|---|---|---|
| Parameters | 10.7M | 25.2M | 10.7M | 25.2M |
| Training (one T4) | 3.93B tokens, 8.6 h | 1.97B tokens, 9.1 h | +413M tokens, 56 min | +413M tokens, 1.85 h |
| Val loss (nats/char) | 0.386 | 0.366 | 0.393 | 0.374 |
| Legal-move rate | 95.6% | 97.7% | 96.1% | **97.8%** |
| Same move as the human | 40.7% | 44.2% | 41.6% | **44.4%** |
| Puzzles solved (1,000) | 8.0% | **11.3%** | 6.8% | 8.3% |
| Score vs Stockfish Skill 0 / 1 / 2 | 0.06 / 0.00 / 0.03 | **0.17** / 0.02 / 0.00 | 0.09 / 0.02 / 0.01 | 0.08 / **0.06 / 0.02** |
| Forfeits vs Stockfish (of 150 games) | 49 | 39 | 52 | **28** |

**Legal-move rate by move number** (greedy first try, 100 held-out games, 8,197 positions):

| Moves | 1–10 | 11–20 | 21–30 | 31–40 | 41–50 | 51–60 | 61–70 | 71–80 | 81–90 |
|---|---|---|---|---|---|---|---|---|---|
| base | 99.8 | 97.0 | 94.9 | 92.2 | 91.3 | 93.8 | 95.5 | 89.1 | 85.0 |
| large | 99.9 | 99.0 | 96.2 | 96.1 | 95.5 | 97.1 | 97.0 | 97.0 | 95.0 |
| ft-base | 99.8 | 97.2 | 94.7 | 93.2 | 91.7 | 97.9 | 96.5 | 90.1 | 87.5 |
| ft-large | 99.9 | 98.7 | 96.8 | 96.1 | 96.1 | 97.4 | 97.5 | 96.0 | 92.5 |

**Puzzles by rating** (% solved, 200 Lichess puzzles per bucket):

| Rating | <1000 | 1000–1499 | 1500–1999 | 2000–2499 | 2500+ |
|---|---|---|---|---|---|
| base | 19.0 | 10.5 | 5.0 | 3.0 | 2.5 |
| large | 26.0 | 16.0 | 8.0 | 5.0 | 1.5 |
| ft-base | 13.0 | 9.5 | 6.5 | 4.0 | 1.0 |
| ft-large | 15.5 | 11.5 | 6.5 | 4.0 | 4.0 |

How these were measured:
- **Legal move** = the model's text is exactly the SAN of a legal move ([decision 008](docs/decisions/008-legal-means-exact-san.md)). "Same move as the human" is on the same positions.
- **Stockfish** = Stockfish 19 at UCI Skill Level 0, 1 and 2, 50,000 nodes per move, 50 games per level with alternating colours ([decision 011](docs/decisions/011-stockfish-levels.md)). Score = (wins + ½ draws) / games. The model gets 6 tries per move (greedy, then 5 sampled); after 6 illegal outputs it **forfeits** the game. With only 50 games per level, score differences of a few hundredths are noise.
- **Val loss** is on the original validation set (games from the same month as training).

What the results say:
- **Model size ([experiment 009](docs/decisions/009-experiment-model-size.md)).** Given the same compute (one ~9 h session), the 25.2M model beats the 10.7M one on every metric even though it saw half as many tokens: 47% fewer illegal outputs, puzzles 8.0 → 11.3%, and it holds its legality late in the game (95% at moves 81–90 vs 85%). Neither model overfit (train ≈ val loss throughout).
- **Fine-tuning on long games ([decision 012](docs/decisions/012-finetune-long-games.md)).** One pass over 594k games of 100+ plies from another month, aimed at the late-game legality drop. It is a trade-off. Both models forgot a little (val loss +0.008) and solved fewer puzzles. Only the large one gained where it was aimed: forfeits fell from 39 to 28 and it has the best legal-move rate. The small one gained nothing (49 → 52 forfeits). Likely reasons: long games are more often drawn (16.9% vs 5.8%), so they hold fewer decisive tactics, and the small model has less spare capacity.

**First test run** (pipeline check, not a result): 30 minutes on a T4 with a 20k-game sample (8M tokens). Val loss went from 3.70 to 0.574 at 147M tokens, then rose as the model started memorising the small sample.

## Repository layout

| Path | Contents |
|------|----------|
| `configs/` | `data.yaml` (download, filters, split), `base.yaml` (model and training) |
| `scripts/` | `download_lichess.py`, `build_dataset.py`, `plot_curves.py`, `export_onnx.py`, `make_parity_fixtures.py` |
| `chessgpt/` | `movetext.py`, `tokenizer.py`, `model.py`, `train.py`, `player.py` |
| `eval/` | `metrics.py` (legal-move rate, puzzles, score vs engine); `vs_stockfish.py`, `puzzles.py` (W4) |
| `kaggle/` | thin training notebook + CLI scripts ([kaggle/README.md](kaggle/README.md)) |
| `web/` | the browser demo ([web/README.md](web/README.md)) |
| `tests/`, `web/test/` | pytest suite, Node tests |
| `docs/decisions/` | decision log: why each data and format choice was made |
| `JOURNAL.md` | weekly notes |

## How to run

Setup (Python ≥ 3.11, [uv](https://docs.astral.sh/uv/)):

```bash
uv sync                      # .venv with CPU-only torch and the dev tools
uv run pytest                # Python tests
```

Data:

```bash
uv run python scripts/download_lichess.py --max-games 20000   # quick sample; configs/data.yaml sets the real run
uv run python scripts/build_dataset.py --sanity               # train.bin, val.bin, meta.json + legality check
```

Training (on Kaggle, see [kaggle/README.md](kaggle/README.md)):

```bash
kaggle/push_dataset.sh data/processed "message"   # private Kaggle dataset
kaggle/run.sh --name base --hours 11              # needs the commit pushed to GitHub
kaggle/fetch_outputs.sh base
uv run python scripts/plot_curves.py runs/kaggle/<run>/runs/base/log.csv
```

`python -m chessgpt.train --config configs/base.yaml --data data/processed --out runs/x` also runs locally (CPU, slow); `--resume`, `--max-hours` and `--iters-this-run` let a run span several sessions.

Web demo (see [web/README.md](web/README.md)):

```bash
uv run python scripts/export_onnx.py runs/<run>/best.pt --out web/models/base   # then list it in web/models/models.json
cd web && python -m http.server 8765                                           # open http://localhost:8765/
```

Design decisions and their reasons are in [docs/decisions/](docs/decisions/).
