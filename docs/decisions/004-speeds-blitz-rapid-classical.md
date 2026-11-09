# 004: Keep blitz, rapid and classical games; exclude bullet

- **Date:** 2026-11-09
- **Status:** accepted

## Context
The time control affects move quality: faster games mean more time-pressure blunders.
The filter reads the PGN `Event` header (`"Rated Blitz game"` → blitz). Variants and
correspondence were already out of scope (brief). In the 20k-game sample (≥1500): blitz 15,822
(79%), rapid 4,036 (20%), classical 142 (<1%). Speed was the largest rejection reason: 21,450 of
57,880 games scanned, mostly bullet.

## Options
1. **Blitz + rapid + classical**: the most data with reasonable quality; blitz dominates.
2. **Rapid + classical only**: better-considered moves, but about 5× less data per game scanned,
   so collection is about 5× slower.
3. **Also add bullet**: roughly doubles the data rate, but bullet is full of time-pressure blunders
   and pre-moves.

## Decision
Blitz, rapid and classical (`speeds: [blitz, rapid, classical]`).

## Why
Blitz above the rating floor is decent chess and is where most of the data is; dropping it would
make collection impractically slow. Bullet's move quality is too poor to be worth imitating.

## Consequences
- The dataset is mostly blitz (~80%), so the model mostly imitates blitz-quality play.
- The mix was measured at the 1500 floor; with `min_elo: 1800` (003), re-check it in
  `games.stats.json` after the full download.

## Follow-up (2026-11-09)
At the 1800 floor (1.79M games): blitz 82.0%, rapid 17.4%, classical 0.6%, so the mix shifted
slightly further toward blitz than in the 1500 sample (79 / 20 / <1). Results: 1-0 49.0%,
0-1 45.2%, draws 5.8%.
