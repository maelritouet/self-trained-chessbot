# web/

A static page to play the bot in the browser. No build step: plain ES modules, with
ONNX Runtime Web and chess.js loaded from jsDelivr at pinned versions.

| File | Role | Tested by |
|------|------|-----------|
| `js/player.js` | JS port of `chessgpt/player.py`'s `choose_move` (prompt, sampling, exact-SAN legality, retries, top-k scoring, `onEvent` progress hook) | `test/player.test.mjs`, incl. parity with Python on `test/fixtures/` |
| `js/game.js` | Game state: turns, legal targets, promotion, end of game, fallback move | `test/game.test.mjs` |
| `js/app.js` | DOM wiring: board, controls, model loading, the "pen" strip | by playing it (see below) |
| `index.html`, `css/style.css` | The page | |

## Plug in a model

```bash
python scripts/export_onnx.py runs/<run>/best.pt --out web/models/<name>
```

then list it in `web/models/models.json` (first entry loads by default; a selector appears
when there are two or more, e.g. base vs fine-tuned):

```json
[
  {"id": "base", "label": "Base model"},
  {"id": "finetuned", "label": "Fine-tuned on 2000+ games"}
]
```

`web/models/` is gitignored (generated files, ~50 MB per fp32 model).

## Run locally

```bash
cd web && python -m http.server 8765    # then open http://localhost:8765/
```

(ES modules and `fetch` need a server; opening `index.html` as a file won't work.)

## Tests

```bash
export PATH=$PWD/tools/node/bin:$PATH   # Node LTS lives in tools/ (gitignored)
cd web && npm install && npm test
```

To regenerate the parity fixtures after changing the Python player:

```bash
python -m chessgpt.train --config web/test/fixtures/tiny.yaml --data data/processed --out runs/parity-tiny
python scripts/make_parity_fixtures.py runs/parity-tiny/best.pt
```

## Checked so far

28 Node tests pass (16 for the player, incl. parity with Python on 14 positions; 12 for the game
state). The page was played in headless Chromium (Playwright): a 25-move game with the 30-minute
test model, no console errors, illegal attempts crossed out, two fallback moves marked `*`.

## Known limits (for W6)

- Speed: every legal move is scored over the whole prompt, with no cache between calls.
  Measured in Chromium (single-threaded WASM): 0.1–0.6 s per reply up to move 11, then 1.4–9 s.
- Size: the fp32 model is ~50 MB; fp16 or int8 quantisation would cut it 2-4×.
- Multi-threaded WASM needs cross-origin isolation headers, which GitHub Pages can't set, so
  it runs single-threaded there.
