# 007: Build the training set from 3M games

- **Date:** 2026-11-09
- **Status:** accepted

## Context
How much data to collect depends on how many tokens training will actually consume. The first
Kaggle test run (30 min on a T4, 20k-game sample of 8.01M train tokens, `configs/base.yaml`,
10.7M parameters) measured:
- throughput ~100k tokens/s including evaluations, i.e. ~4B tokens per 11 h session;
- best val loss 0.574 at 147M tokens seen, then val loss rising while train loss kept falling:
  the model started memorising after ~18 passes over its data.

At ~405 characters per game, 1M games ≈ 0.4B tokens. The current `base.yaml` run
(20,000 iterations × 32,768 tokens) consumes 655M tokens, about 1.8 h on the T4. For reference,
the Chinchilla rule of thumb (~20 tokens per parameter) gives ~214M tokens for 10.7M parameters.

## Options
1. **1M games** (~0.4B tokens, `train.bin` ~0.4 GB): enough for one `base.yaml` run (~1.6 passes),
   but a full 11 h session would repeat it ~10×, bringing overfitting back, especially for a 25M model.
2. **3M games** (~1.2B tokens, ~1.2 GB): ~0.5 passes for `base.yaml`, ≤ ~3.3 passes even for a full
   11 h session. Already being downloaded at the 1800 floor (~1.5 h).
3. **6M+ games** (≥2.4B tokens, ≥2.4 GB): a full session barely repeats data, but ~5 h of download
   and a larger upload; only pays off for long runs or a bigger model.

## Decision
3M games (`max_games: 3000000`).

## Why
It removes the overfitting seen in the test run by a wide margin: the planned run sees each game about
half a time, and even the longest possible session stays far below the ~18 passes where memorisation
began. It is already downloading, so it costs no extra time, and a 1.2 GB upload is manageable.

## Consequences
- `base.yaml` uses about half the data; there is room for longer runs in W3 without repeating much.
- If W3 moves to longer runs or the 25M-parameter model, revisit: more data can come from an earlier
  month (e.g. 2026-09) with the same filters, appended before rebuilding.
- The 18-pass overfitting point was measured on a tiny dataset; on 1.2B tokens the safe number of
  passes may differ. Watch the train/val gap in `log.csv`.
