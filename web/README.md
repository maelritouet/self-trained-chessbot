# web/

A static page to play the bot in the browser. No build step: plain ES modules, with
ONNX Runtime Web and chess.js loaded from jsDelivr at pinned versions.

| File | Role | Tested by |
|------|------|-----------|
| `js/player.js` | JS port of `chessgpt/player.py`'s `choose_move` (prompt, sampling, exact-SAN legality, retries, top-k scoring, `onEvent` progress hook) | `test/player.test.mjs`, incl. parity with Python on `test/fixtures/` |
| `js/game.js` | Game state: turns, legal targets, promotion, end of game, fallback move | `test/game.test.mjs` |
| `js/app.js` | DOM wiring: board, controls, model loading, the "pen" strip | by playing it (see below) |
| `index.html`, `css/style.css` | The page | |

## Get the models

The four models (base, large and their fine-tunes, ~320 MB in all) are attached to the
GitHub release `v1.0` (decision 013). From the repo root:

```bash
python3 scripts/fetch_models.py               # all four, checked against web/models.release.json
python3 scripts/fetch_models.py --only base   # or just one (49 MB)
```

This fills `web/models/<id>/{model.onnx,meta.json}` and writes `web/models/models.json`, the
model menu (first entry loads by default). `web/models/` is gitignored.

To add your own model instead, export a checkpoint and list it in `models.json` by hand:

```bash
python scripts/export_onnx.py runs/<run>/best.pt --out web/models/<name>
```

## Run locally

```bash
cd web && python3 -m http.server 8765    # then open http://localhost:8765/
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
state). The page was played in headless Chromium (Playwright) with all four release models:
no console errors, illegal attempts crossed out, fallback moves marked `*`, no horizontal
scroll at phone width (390 px).

## Known limits

- Speed: every legal move is scored over the whole prompt, with no cache between calls. In
  Chromium (single-threaded WASM), base replies in 0.1–0.6 s up to move 11, then 1.4–9 s;
  large takes about twice as long.
- Size: the models are fp32 (49 MB base, 112 MB large); fp16 would halve that but needs its
  own parity check.
- Multi-threaded WASM needs cross-origin isolation headers, which `python -m http.server`
  doesn't send, so it runs on one thread. `app.js` uses up to 4 threads when the page is
  isolated.
