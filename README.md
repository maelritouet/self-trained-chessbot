# self-trained-chessbot

A small decoder-only transformer (~10–25M parameters) trained **from scratch** on Lichess game records. It is never told the rules of chess: it only learns to predict the next character of a game's movetext (`1-0 1.e4 e5 2.Nf3 Nc6 …`). [python-chess](https://python-chess.readthedocs.io/) acts as the referee.

## Final product

1. **The trained model + a results table:** legal-move rate (overall and by move number), Lichess puzzle accuracy by rating, score vs Stockfish levels 1–3, one controlled experiment, and before/after fine-tuning.
2. **A self-hosted website to play against it:** the model runs in the browser (ONNX Runtime Web), chess.js checks legality, and the page shows the model's top-5 candidate moves with probabilities, its illegal attempts, a temperature slider and a base/fine-tuned toggle.

## How it works

- **Data.** Rated blitz, rapid and classical games where both players are rated 1800+, streamed from the [Lichess database](https://database.lichess.org) (2026-10 dump) and rewritten as one line per game: `1-0 1.e4 e5 2.Nf3 …`. The result comes first, so at play time the model can be asked to play like the side that won.
- **Tokens.** One token per character: 33 symbols (SAN letters, digits, `+#=-/.`, space, and a newline between games), stored as `uint8` arrays nanoGPT-style.
- **Model.** GPT-2-style decoder: 6 layers, 6 heads, 384-dim embeddings, 1,024-character context, 10.7M parameters.
- **Playing.** To choose a move, the model writes characters until a space. The text counts as a move only if it is **exactly** the SAN of a legal move; otherwise the attempt is recorded and it tries again. Separately, every legal move is scored by the model's probability of writing it, which gives the top-5 list.
- **Training.** On Kaggle (one T4 GPU, fp16 mixed precision, ~100k tokens/s), driven from a laptop with the Kaggle CLI.

## Status

| Week | Focus | Status |
|------|-------|--------|
| W0 | Setup | done |
| W1 | Data pipeline + tokenizer | done; 3M-game dataset being built |
| W2 | Model, training loop, Kaggle pipeline | done; first 30-min test run on a 20k-game sample |
| W3 | Training discipline + experiment | next: first real run on the 3M-game dataset |
| W4 | Play + eval harness | metrics implemented; Stockfish games and puzzles to do |
| W5 | Fine-tune + write-up | — |
| W6 | Web demo | working locally with the test model; speed, size and hosting to do |

## Results

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Legal-move rate (overall) | | |
| Legal-move rate (moves 30+) | | |
| Puzzle accuracy (<1500 / 1500–2000 / 2000+) | | |
| Score vs Stockfish L1 / L2 / L3 | | |

**First test run** (pipeline check, not a result): 30 minutes on a T4 with a 20k-game sample (8M tokens). Val loss went from 3.70 to 0.574 at 147M tokens, then rose as the model started memorising the small sample. On 20 held-out games it played a legal first greedy move in 81% of positions (96% in moves 1–10, 63% by moves 41–50), measured before move legality was made strict (exact SAN only).

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
