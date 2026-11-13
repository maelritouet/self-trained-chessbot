# 014: A public demo: the page on GitHub Pages, the models on the Hugging Face Hub

- **Date:** 2026-11-13
- **Status:** accepted (supersedes the "local only" part of 013)

## Context
Decision 013 kept the demo local: a reader had to clone the repo, download ~320 MB and start a
server to play, so most readers only saw screenshots. A live link makes the project much easier
to show. The page is already static (plain ES modules; ONNX Runtime Web and chess.js from
jsDelivr at pinned versions), so only the models need a host.
- GitHub Pages serves the page but caps files at 100 MB (large is 112 MB) and the models stay
  out of git anyway.
- GitHub release files can't be fetched by a page on another site (no CORS headers).
- Hugging Face model repos serve large files with CORS headers from a CDN, for free.

## Options
1. **Keep 013 (local only).** Nothing new to maintain, but no public link.
2. **Hugging Face static Space for both page and models.** One public link, and it could send the
   headers that enable multi-threaded WASM. The page would then live outside the GitHub repo
   and need a separate deploy.
3. **GitHub Pages for the page, a Hugging Face model repo for the models.** The page deploys from
   this repo on every push to main; the models get a proper model card. Two places to keep in
   sync, single-threaded WASM.
4. **GitHub Pages with fp16 base models only.** No second service, but drops the large models,
   which are the best ones.

## Decision
Option 3. The page is deployed from `web/` by a GitHub Actions workflow. The four models are in
the public model repo `Mrenaissance/self-trained-chessbot`, one folder per model (`base/`,
`large/`, `ft-base/`, `ft-large/`, each with `model.onnx` and `meta.json`) plus a model card.
The page downloads from `https://huggingface.co/<repo>/resolve/main/<model>/…`, checks each
file's size and SHA-256 against `web/models.release.json`, and keeps it in the browser's Cache
Storage so it downloads only once. Base loads first (49 MB) so the page is playable quickly.

The local path from 013 stays: `scripts/fetch_models.py` still downloads the same files from the
GitHub release, and the page uses them if Hugging Face can't be reached, or always with `?local`.

The models stay fp32. Storing the weights as fp16 (fp32 arithmetic) would cut the downloads
from 51/112 MB to 24/53 MB and played the same greedy move as fp32 on all 14 parity positions
for every model, but candidate probabilities moved by up to 0.4% relative, above the 0.01% the
JS/Python parity test allows. `scripts/fp16_onnx.py` makes those copies if that trade-off is
ever wanted.

## Why
- A link anyone can open beats screenshots.
- The repo remains the single source of truth: the page, the manifest with checksums and the
  release all live in it, and Hugging Face only hosts the same bytes. Since the files are
  identical to the v1.0 release, one manifest checks both sources.
- Pages deploys from the repo on every push to main, so the live page always matches the code.

## Consequences
- Two places to update when models change: the release (for `fetch_models.py`) and the Hugging
  Face repo. A checksum mismatch makes a forgotten update fail loudly instead of silently.
- Single-threaded WASM on Pages (no cross-origin isolation headers): replies take 0.1–9 s.
- The page depends on huggingface.co and jsDelivr being up; offline use goes through `?local`.
- Revisit fp16 if the download size becomes the main complaint.
