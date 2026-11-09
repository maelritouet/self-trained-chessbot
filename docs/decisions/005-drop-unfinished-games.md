# 005: Drop games without a real result

- **Date:** 2026-11-09
- **Status:** accepted

## Context
PGN allows `Result "*"` (unknown or unfinished). With the result placed first in the movetext
(001), such a game would start with `*` instead of a real outcome. In the 20k-game sample,
0 of 57,880 scanned games had `*`, so this is a guard rather than a filter that removes data today.

## Options
1. **Drop them** (`no_result` rejection in `download_lichess.py`).
2. **Keep them**, adding `*` to the vocabulary as a fourth "result".

## Decision
Drop them.

## Why
A result-conditioned model needs every game's first token to be a real outcome; `*` would be a
meaningless condition. It costs essentially no data.

## Consequences
- Every training game starts with `1-0`, `0-1` or `1/2-1/2`, which `choose_move`'s
  `result_prefix` relies on.
- The count shows up as `rejected:no_result` in the download stats; if it ever becomes
  non-negligible, investigate why rather than keeping the games.
