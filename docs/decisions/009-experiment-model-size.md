# 009: The controlled experiment varies model size (10.7M vs 25.2M params)

- **Date:** 2026-11-10
- **Status:** accepted

## Context
The project needs one controlled experiment: change one thing, keep the rest, measure. The base
run (`configs/base_11h.yaml`: 6 layers x 384 wide, 10.7M params excluding position embeddings,
3.93B tokens = 3.15 passes over 3M games) ended at val loss 0.3856 with train 0.3837. That's
no overfitting, and the loss was still falling slowly at the minimum learning rate, which
suggests the model, not the data, is the limit.

## Options
1. **Model size**: same data, one bigger model.
2. **Result prefix vs none**: tests decision 001. Needs a rebuilt dataset and re-upload.
3. **Block size** (512 vs 1024): 99.5% of games already fit in 1024, so the effect is likely small.
4. **Data amount** (1M vs 3M games): the base run shows no overfitting at 3M, so less data
   would mostly show a curve we can already predict.

## Decision
Model size. `configs/large_11h.yaml`: 8 layers x 8 heads x 512 wide = 25.2M params (25.8M with
position embeddings), head size 64 as in the base model. Everything else is identical: data,
tokenizer, batch 32 x 1024, learning rate 6e-4 → 6e-5 cosine, warmup, weight decay, seed.

Budget: **equal compute, not equal tokens.** The large model costs ~2.2x the training FLOPs per
token (~202 vs ~92 MFLOP), so one ~11 h T4 session fits ~2B tokens: 60,000 iterations = 1.97B
tokens (~0.6 of the base run's tokens, ~9.5 h estimated vs the base run's 8.6 h).

## Why
- Size is the variable today's curve points at, and it needs no new data.
- Equal compute answers the practical question ("what is the best model for one Kaggle
  session?") in a single session, without relying on the still-untested cross-session resume.
- An equal-token comparison comes almost for free: the base run's `log.csv` has val loss at
  1.97B tokens (0.4182 at iteration 60,000). Caveat: at that point the base run's learning rate
  was still 3.3e-4 (cosine half-way, not yet annealed), which flatters the large model. Report
  it as secondary.

## Consequences
- Compare at the end of each run on the same metrics: val loss, strict legal-move rate on the
  same held-out games (overall and by move number), puzzles by rating, score vs Stockfish.
- If the T4 is slower than estimated, `--max-hours` stops the run before the cosine ends and
  the comparison is no longer clean. Check `stopping at iter` in the notebook log.
- The winner becomes the model that is fine-tuned in W5 and served on the web page.
