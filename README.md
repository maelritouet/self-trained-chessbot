# self-trained-chessbot

A small decoder-only transformer (~10–25M parameters) trained **from scratch** on Lichess game records. It is never told the rules of chess: it only learns to predict the next character of a game's movetext (`1.e4 e5 2.Nf3 Nc6 …`). [python-chess](https://python-chess.readthedocs.io/) acts as the referee.

## Final product

1. **The trained model + a results table:** legal-move rate (overall and by move number), Lichess puzzle accuracy by rating, score vs Stockfish levels 1–3, one controlled experiment, and before/after fine-tuning.
2. **A self-hosted website to play against it:** the model runs in the browser (ONNX Runtime Web), chess.js checks legality, and the page shows the model's top-5 candidate moves with probabilities, a temperature slider and a base/fine-tuned toggle.

## Status

| Week | Focus | Status |
|------|-------|--------|
| W0 | Setup | in progress |
| W1 | Data pipeline + tokenizer | in progress |
| W2 | Model | — |
| W3 | Training discipline + experiment | — |
| W4 | Play + eval harness | — |
| W5 | Fine-tune + write-up | — |
| W6 | Web demo | — |

## Results

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Legal-move rate (overall) | | |
| Legal-move rate (moves 30+) | | |
| Puzzle accuracy (<1500 / 1500–2000 / 2000+) | | |
| Score vs Stockfish L1 / L2 / L3 | | |

## How to run

```bash
uv sync                                           # create .venv (CPU torch)
uv run python scripts/download_lichess.py --max-games 20000
uv run python scripts/build_dataset.py --sanity
uv run pytest
```

Training runs on Kaggle (see `kaggle/`, added in W2). Settings are in `configs/`, and design decisions are logged in `docs/decisions/`.
