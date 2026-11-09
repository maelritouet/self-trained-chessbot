# 001: Put the game result first in the movetext

- **Date:** 2026-11-09
- **Status:** accepted

## Context
Every Lichess game has a result (`1-0`, `0-1`, `1/2-1/2`). The model only sees movetext, so we
had to decide whether the result appears in the training text at all, and where.

## Options
1. **Drop it**: `1.e4 e5 … Qxf7#`. The model only ever predicts moves. Simplest, but it never
   learns when a game is "over", and a free signal is thrown away.
2. **Suffix**: `1.e4 e5 … Qxf7# 1-0`. The model learns to predict outcomes from positions.
   `choose_move` must recognise and reject a result token sampled mid-game.
3. **Prefix**: `1-0 1.e4 e5 …`. Conditioning: at play time the prompt starts with the result we
   want the model's side to achieve, so it imitates the winning side's moves (the idea used in
   Karvonen's Chess-GPT work).

## Decision
Prefix (`result_position: prefix` in `configs/data.yaml`).

## Why
It is the only option that can make the bot *stronger* at play time for free: asking for
`1-0` when playing White shifts the model toward moves made by players who went on to win.
It also keeps the outcome information in the data instead of discarding it.

## Consequences
- Every training game starts `\n<result> 1.`; prompts must be built the same way
  (`chessgpt.player.build_prompt`, `result_prefix` argument). Passing no prefix is out of distribution.
- Adds `/` to the vocabulary (from `1/2-1/2`); `-`, `0`, `1`, `2` were already present.
- In games longer than the context (`block_size`, see 006), the prefix falls out of the window,
  so late moves lose the conditioning.
- `result_position` stays configurable, so prefix vs none is a ready-made candidate for the W3
  controlled experiment. Revisit if conditioning shows no measurable effect on play strength.
