# web/

A static page to play the bot in the browser. No build step: plain ES modules, with
ONNX Runtime Web and chess.js loaded from jsDelivr at pinned versions.

| File | Role | Tested by |
|------|------|-----------|
| `js/player.js` | JS port of `chessgpt/player.py`'s `choose_move` (prompt, sampling, exact-SAN legality, retries, top-k scoring, `onEvent` progress hook) | `test/player.test.mjs`, incl. parity with Python on `test/fixtures/` |
| `js/game.js` | Game state: turns, legal targets, promotion, end of game, fallback move | `test/game.test.mjs` |
| `js/app.js` | DOM wiring: board, controls, model loading, the "pen" strip | by playing it (see below) |
| `index.html`, `css/style.css` | The page | |

## Where the models come from

`models.release.json` (committed) is the model menu, in order (the first loads by default), with
each file's size and SHA-256. The page (decision 014):

1. downloads `https://huggingface.co/<hf_repo>/resolve/main/<id>/{meta.json,model.onnx}`
   (the public model repo `Mrenaissance/self-trained-chessbot`), with a progress bar;
2. checks size and SHA-256 (`crypto.subtle`, so only on https or localhost);
3. keeps the checked bytes in Cache Storage (`chessbot-models-v1`), keyed by URL and checksum,
   so the next visit starts without a download and a new model version is a new key;
4. if Hugging Face can't be reached, tries the same model from `models/<id>/`.

With `?local` in the URL it skips Hugging Face and lists the models in `models/models.json`.

The deployed page is `index.html`, `css/`, `js/` and `models.release.json`, copied by
`.github/workflows/pages.yml` on every push to main that touches `web/`.

## Run locally (offline, or with your own model)

The same four models (~320 MB) are attached to the GitHub release `v1.0` (decision 013). From the
repo root:

```bash
python3 scripts/fetch_models.py               # all four, checked against web/models.release.json
python3 scripts/fetch_models.py --only base   # or just one (51 MB)
cd web && python3 -m http.server 8765         # then open http://localhost:8765/?local
```

This fills `web/models/<id>/{model.onnx,meta.json}` and writes `web/models/models.json`.
`web/models/` is gitignored. To add your own model, export a checkpoint and list it in
`models.json` by hand (models not in the manifest load without a checksum):

```bash
python scripts/export_onnx.py runs/<run>/best.pt --out web/models/<name>
```

(ES modules and `fetch` need a server; opening `index.html` as a file won't work.)

## Publish new models

```bash
python scripts/fetch_models.py --stage dist/release --tag v1.1 base="Base (10.7M params)" ...
python scripts/push_hf.py          # uploads web/models/<id>/ and docs/model_card.md (hf auth login first)
```

Upload the staged files to the GitHub release too, so both sources match the manifest.

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
- Size: the models are fp32 (51 MB base, 112 MB large). `scripts/fp16_onnx.py` makes
  fp16-weight copies (24 / 53 MB): on the 14 parity positions every model played the same greedy
  move, but probabilities moved by up to 0.4% relative, above the parity test's 0.01%, so they
  aren't used.
- Multi-threaded WASM needs cross-origin isolation headers, which neither GitHub Pages nor
  `python -m http.server` sends, so it runs on one thread. `app.js` uses up to 4 threads when the page is
  isolated.
