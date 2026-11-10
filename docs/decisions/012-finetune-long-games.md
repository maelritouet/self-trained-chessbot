# 012: Fine-tune on long games, to fix late-game legality

- **Date:** 2026-11-10
- **Status:** accepted

## Context
The base model's evals (2026-11-10) point to one weakness:
- The legal-move rate falls with game length: 99.8% at moves 1-10, ~92% at 31-50, 85-89% past
  move 70 (100 val games, 8,197 positions).
- Against Stockfish skill 0/1/2, 24%, 30% and 44% of games end in a **forfeit** (no legal move
  in 6 tries). Forfeits cause 25-45% of all losses.
- Late positions are rare in training: in those 100 games there are 1,995 positions at moves 1-10
  but 101 at moves 71-80. The base data has every opening many times and each endgame once.

## Options
1. **Long games**: fine-tune on games of at least 100 plies, which targets the measured weakness.
2. **Strong players** (both ≥ 2200): aims at move quality. But the score is capped by forfeits
   first, and stronger players' games aren't longer.
3. **Self-play / rejection sampling**: generate games with the model and keep the legal ones.
   Interesting, but the data would be its own, and it needs a new pipeline.
4. **No fine-tune, train longer**: the experiment (009) already covers "more compute".

## Decision
Option 1.
- **Data:** `configs/data_finetune.yaml` = `data.yaml` with three changes. The month is 2026-09,
  never seen by the base model (trained on 2026-10). `min_plies` is 100. `max_games` is 600,000
  (~300M tokens). Same vocabulary (`build_dataset.py --meta data/processed/meta.json`).
- **Training:** start from the weights of the model that wins experiment 009, with a fresh optimizer
  (`--init-from`), LR 1e-4 → 1e-5 cosine, ~1 pass over the data.
- **Before/after:** the same evals (legal rate on the same 100 val games, by move number, error
  kinds; puzzles; Stockfish 0/1/2), plus val loss on the **original** val set, to check that
  the model hasn't forgotten its openings.

## Why
- It aims at what the evals measured, and the before/after table can show it: legality past move
  30, forfeit counts.
- The low LR and a single pass keep it a fine-tune, not a second training run.
- A new month means the improvement can't come from seeing the same games again.

## Consequences
- The fine-tune waits for experiment 009 to pick its starting model (the `large` run ends tonight).
- Risk: long games are more often drawn, slow endgames. The result-prefix mix shifts toward
  `1/2-1/2`, which may make the model more drawish when prompted with a draw. Check the result
  counts in the built dataset.
- If val loss on the original val set rises clearly, lower the LR or mix in normal games.
