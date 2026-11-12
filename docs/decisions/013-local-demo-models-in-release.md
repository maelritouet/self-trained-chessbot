# 013: The web demo runs locally; the models ship as a GitHub release

- **Date:** 2026-11-12
- **Status:** accepted

## Context
The project goal said "a self-hosted website to play the bot". The page works locally with all
four models (base, large, ft-base, ft-large; decision 009 ships both sizes). The exported ONNX
files are 49 MB (base, ft-base) and 112 MB (large, ft-large), ~320 MB in all, fp32.
- Git refuses them: GitHub rejects files over 100 MB, and generated weights stay out of git
  by design.
- GitHub Pages has the same 100 MB file limit. GitHub release files can't be fetched by a page on
  another site (no CORS headers). Cloudflare Pages caps files at 25 MB.

## Options
1. **Run locally, models in a GitHub release**: clone, run one script that downloads the models,
   start a local server. Free, one place (the repo), no account elsewhere. Visitors must clone.
2. **Hugging Face static Space**: a public link, large files fine, can enable multi-threaded
   WASM. Needs an HF account and a second place to keep in sync.
3. **GitHub Pages + models on Hugging Face**: a public link, single-threaded, two places.
4. **GitHub Pages with fp16 base models only**: a public link, but drops the better large models.

## Decision
Option 1. The four models are attached to a GitHub release (`v1.0`). `scripts/fetch_models.py`
downloads them into `web/models/`, checks each file's SHA-256 against `web/models.release.json`
(committed), and writes `web/models/models.json`. The README tells the story with results,
curves and screenshots of the page.

## Why
- The repo is the portfolio: everything (code, results, journey, demo) stays in one place.
- No second service to maintain or pay for; release files have a 2 GB limit, far above 112 MB.
- Checksums make a truncated or wrong download fail loudly instead of loading a broken model.

## Consequences
- No public link: a reader sees the demo through screenshots unless they clone and run it.
- The models are fp32. fp16 would halve the download but needs its own parity check; not done.
- Revisit if a live link is wanted: the same files can go to a Hugging Face Space (option 2)
  without changing the page.
