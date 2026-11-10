# 010: Puzzles get their move history from the Lichess game export API

- **Date:** 2026-11-10
- **Status:** accepted

## Context
The Lichess puzzle database (`lichess_db_puzzle.csv.zst`) gives each puzzle as a FEN plus a UCI
move list (the first move is the opponent's setup move). Our model has no FEN input: it reads
movetext from move 1 (`1-0 1.e4 e5 …`). So each puzzle needs the real game history up to the
puzzle position. Every row has a `GameUrl` (`https://lichess.org/<id>/black#48`).

## Options
1. **Lichess game export API**: fetch the source games' PGNs for a sample of puzzles and cache
   them. Exact history for any puzzle.
2. **Match our own dump**: keep only puzzles whose game is in the 2026-10 dump we downloaded.
   No network, but small overlap, biased to 1800+ games of one month.
3. **FEN prompt**: not possible without a new input format and retraining.

## Decision
Option 1. A fixed sample, stratified by puzzle rating with the bucket edges `puzzle_accuracy`
uses (<1000, 1000-1499, 1500-1999, 2000-2499, 2500+), 200 puzzles per bucket, 1,000 in total,
drawn with a fixed seed. Only puzzles with a settled rating (`RatingDeviation` ≤ 90) are used.
Games are fetched in batches of up to 300 IDs with `POST /api/games/export/_ids`, one request
at a time, waiting on HTTP 429. The result is cached in `data/puzzles/` (outside git).

## Why
- Exact histories for the whole rating range, which is what "accuracy by rating" needs.
- Cheap: 1,000 games is 4 requests.
- A sanity check makes it safe: replaying the history must reproduce the puzzle's FEN exactly
  (board, side to move, castling, en passant), or the puzzle is dropped and counted.

## Consequences
- The puzzle set is a file anyone can rebuild with the same seed. The script logs how many
  puzzles were dropped and why.
- Source games can be any rating or speed (puzzles come from all of Lichess), unlike training
  games (1800+, no bullet). This is part of what the metric measures.
- 200 puzzles per bucket gives a ±7-point 95% interval at 50% accuracy. Enough to show a
  trend across ratings, not to separate two close models in one bucket.
