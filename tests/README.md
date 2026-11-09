# tests/

Run with `uv run pytest` (or `.venv/bin/python -m pytest`). The browser code has its own
Node tests in `web/test/` (`cd web && npm test`, see `web/README.md`).

## Written

Written together with the code they test, **test-first**: each test was seen failing
against a stub before the implementation existed, and each file was mutation-checked (deliberate bugs put into the code to confirm a test catches
them). Two weak tests found that way were fixed in their own commits.

| File | Tests | Covers |
|------|------:|--------|
| `test_model.py` | 14 | GPT shapes, loss, initial loss ≈ ln(V), causal masking both ways, manual vs fused attention, dropout in eval mode, hand-derived parameter count, overfitting, `choose_move` integration |
| `test_train.py` | 15 | LR schedule, weight-decay groups, seeded `get_batch`, gradient accumulation, end-to-end run, bit-identical resume, time limit |
| `test_metrics.py` | 20 | position evaluation with an oracle model, legal-move and match rates, move-number buckets, puzzles (incl. an alternative mate), score vs engine with forfeits |
| `test_player.py` | 14 | move legality is exact SAN only (no phantom captures/checks, long algebraic or `--` null move) |
| `test_export_onnx.py` | 6 | ONNX logits match PyTorch for any batch and length, I/O names, `meta.json` |
| `test_build_kernel.py` | 9 | Kaggle notebook parameters and metadata |
| `test_plot_curves.py` | 6 | reading `log.csv`, best val point, PNG output, CLI |

## Still to write

Proposed cases for the data pipeline, which has no tests yet. Suggested file names in brackets.

### Tokenizer [`test_tokenizer.py`]
- Round trip: `decode(encode(s)) == s` for a real game string, the empty string, and a string containing every vocab character.
- `encode_array(s).tolist() == encode(s)` (the fast path matches the slow path).
- Unknown character (e.g. `"Z"`, `"é"`) raises `ValueError` in both `encode` and `encode_array`.
- `from_text` is deterministic: the same character set in any order gives the same `itos`.
- `save` then `load` gives the same `stoi`, `itos` and `vocab_size`.
- Duplicate or multi-character entries in the constructor raise an error.

### Movetext normalisation [`test_normalise.py`] (`scripts/download_lichess.normalise_movetext`, `chessgpt.movetext`)
- Clock and eval comments removed: `1. e4 { [%clk 0:03:00] } 1... e5 { [%eval 0.2] }` → `1.e4 e5`.
- Black move-number markers `1...` removed, including when glued: `1...e5`.
- NAGs removed: `$1`, `$14`; suffix annotations removed: `e4!`, `e4?!`, `Nf3??`, `d4!?`.
- Check, mate, promotion and castling kept intact: `e8=Q+`, `Qxf7#`, `O-O-O`.
- Result placement: `prefix` → `1-0 1.e4 …`, `suffix` → `… 1-0`, `none` → no result; invalid value raises.
- Draw result `1/2-1/2` survives as a single token.
- Ply count is correct for games ending on a white move and on a black move.
- `format_movetext` / `parse_movetext` round trip, with the result as prefix, suffix and absent.
- A game with no moves at all doesn't crash.

### Filters [`test_filters.py`] (`filter_and_normalise`, `event_speed`, `iter_games`)
- `event_speed`: `"Rated Blitz game"` → blitz, `"Rated Rapid tournament https://…"` → rapid, `"Rated Bullet game"` → bullet (rejected), `"Casual Blitz game"` → None.
- Elo: both at exactly `min_elo` → kept; one at `min_elo - 1` → rejected; `"?"` → rejected.
- Termination `"Abandoned"` and `"Rules infraction"` → rejected; `"Normal"` and `"Time forfeit"` → kept.
- `Result "*"` → rejected as `no_result`.
- `min_plies - 1` plies → rejected; exactly `min_plies` → kept.
- Each rejection increments exactly one counter, the first filter that fails.
- `iter_games` splits a two-game PGN string into exactly two `(headers, movetext)` pairs, including a final game with no trailing blank line.

### Dataset split and .bin files [`test_build_dataset.py`] (on a tiny fixture `games.jsonl` in `tmp_path`)
- **No leakage:** every game ends up in exactly one of train or val (decode both `.bin` files, split on the delimiter, check the sets are disjoint and their union equals the input).
- Same seed → identical `train.bin` / `val.bin` bytes; different seed → different split.
- `len(train.bin) + len(val.bin) == sum(len(game) + 1)` (one delimiter per game).
- Every game in the `.bin` starts with the delimiter; decoding `val.bin` gives back the original movetexts.
- A delimiter character inside a movetext raises an error.
- `--meta` reuses the given vocab exactly (same `itos`) even if the new data has fewer characters.

### Remaining player and batch cases (add to the existing files)
- `get_batch`: a `.bin` shorter than `block_size + 2` raises `ValueError` (`test_train.py`).
- `choose_move` on a finished game (e.g. after Fool's mate) raises `ValueError` (`test_player.py`).
- `choose_move` with `temperature=0` is deterministic; a fixed `generator` seed is reproducible (`test_player.py`).
