---
license: cc-by-4.0
library_name: onnx
pipeline_tag: text-generation
tags:
  - chess
  - onnx
  - character-level
  - gpt
  - from-scratch
---

# self-trained-chessbot

Small GPTs (10.7M and 25.2M parameters) trained **from scratch** on 3 million Lichess games. They
are never told the rules of chess: they only learn to predict the next character of a game's
move list (`1-0 1.e4 e5 2.Nf3 Nc6 …`).

- **Play it in your browser:** https://maelritouet.github.io/self-trained-chessbot/
- **Code, training, evaluation and the decision log:** https://github.com/maelritouet/self-trained-chessbot

## Files

One folder per model, each with `model.onnx` (fp32) and `meta.json` (vocabulary, game delimiter,
context size):

| Folder | Model | Size |
|---|---|---|
| `base/` | 6 layers, 6 heads, 384-dim, 10.7M parameters | 51 MB |
| `large/` | 8 layers, 8 heads, 512-dim, 25.2M parameters | 112 MB |
| `ft-base/` | base, fine-tuned on 594k long games | 51 MB |
| `ft-large/` | large, fine-tuned on 594k long games | 112 MB |

The ONNX graph takes `idx`: int64 `(batch, seq)` character ids (seq ≤ 1024) and returns
`logits`: float32 `(batch, seq, 33)`. SHA-256 sums are in
[`web/models.release.json`](https://github.com/maelritouet/self-trained-chessbot/blob/main/web/models.release.json).

## How to use

The prompt is a newline, the result the model's side should achieve, a space, and the movetext so
far, ending where the next move starts: `"\n1-0 1.e4 e5 2."` asks for a white move played to win.
The model writes characters until a space. Its text counts as a move only if it is **exactly** the
SAN of a legal move; otherwise sample again. The full move function (retries, scoring every legal
move) is `chessgpt/player.py` (Python) and `web/js/player.js` (browser) in the GitHub repo.

## Training

- **Data:** rated blitz, rapid and classical games from the [Lichess database](https://database.lichess.org)
  (October 2026, CC0) where both players are rated 1800+, rewritten as one line per game with
  the result first. 3M games, 1.26B characters. The fine-tunes used 594k games of 100+ plies
  from another month.
- **Tokens:** one per character, 33 symbols.
- **Model:** GPT-2-style decoder, 1,024-character context.
- **Compute:** one Kaggle T4 per run, fp16 mixed precision. base: 3.93B tokens in 8.6 h; large:
  1.97B tokens in 9.1 h (the same compute budget, the project's controlled experiment);
  fine-tunes: +413M tokens each.

## Results

All measured on data the models never trained on.

| | base | large | ft-base | ft-large |
|---|---|---|---|---|
| Parameters | 10.7M | 25.2M | 10.7M | 25.2M |
| Val loss (nats/char) | 0.386 | 0.366 | 0.393 | 0.374 |
| Legal-move rate | 95.6% | 97.7% | 96.1% | **97.8%** |
| Same move as the human | 40.7% | 44.2% | 41.6% | **44.4%** |
| Puzzles solved (1,000) | 8.0% | **11.3%** | 6.8% | 8.3% |
| Score vs Stockfish Skill 0 / 1 / 2 | 0.06 / 0.00 / 0.03 | **0.17** / 0.02 / 0.00 | 0.09 / 0.02 / 0.01 | 0.08 / **0.06 / 0.02** |
| Forfeits vs Stockfish (of 150 games) | 49 | 39 | 52 | **28** |

- **Legal move** = the text is exactly the SAN of a legal move, greedy first try, 8,197 positions
  from 100 held-out games.
- **Stockfish** = Stockfish 19 at UCI Skill Level 0, 1 and 2, 50,000 nodes per move, 50 games per
  level. The model gets 6 tries per move and forfeits after 6 illegal outputs.
- **Puzzles** = 1,000 Lichess puzzles, 200 per rating bucket. Lichess rule: every solution move,
  greedy, one try each; any checkmating move also counts.

## Limitations

- No search and no rules: it plays by pattern, and it gets weaker late in long games, where its
  legal-move rate drops (85–95% after move 80).
- It is weak: about 0.17 points per game at best against Stockfish's lowest setting.
- Trained on games between 1800+ players on one site in one month; it has no notion of time
  controls or of who is playing.

## License

The model weights are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The training data (the Lichess database) is CC0.
