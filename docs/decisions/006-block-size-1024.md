# 006: Context length (block_size) of 1,024 characters, keep long games

- **Date:** 2026-11-09
- **Status:** accepted

## Context
`block_size` is how many characters the model can see at once. A game longer than that can't be
seen whole: at play time the oldest moves fall out of the context, including the result prefix
(001). Game lengths in the 20k-game sample (≥1500), including the delimiter: median 378 characters,
90th percentile 652, 99th percentile 913; 24.6% exceed 512, 3.8% exceed 768, 0.40% exceed 1,024,
0.03% exceed 2,048.

## Options
1. **Keep all games, `block_size` 1024**: 99.6% of games fit entirely. Longer games are still
   trained on (random windows slide over them).
2. **Drop games longer than 1024**: everything fits, at a cost of 0.4% of the data, but the model
   never sees very long endgames.
3. **Smaller `block_size` 768**: faster training (attention cost grows with T²), but 3.8% of games
   don't fit and long games lose their opening and prefix sooner.

## Decision
`block_size: 1024`, keeping long games.

## Why
Nearly every game fits in full, so the rare long games are worth keeping as data rather than
dropped. A shorter context would save compute but cut off ten times as many games.

## Consequences
- In the rare games over ~1,000 characters, late moves are played without the start of the game
  (or the result prefix) in context. This is expected to show up as a lower legal-move rate in
  the very late move-number buckets of `legal_rate_by_move_number`, which makes it measurable.
- Training windows start at random offsets, so the model also learns from mid-game windows that
  don't begin at a game start; this is normal nanoGPT-style training.
- Re-check the length distribution on the 1800 dataset (003).
