# 011: "Stockfish level 1-3" means UCI Skill Level 0, 1 and 2

- **Date:** 2026-11-10
- **Status:** accepted

## Context
The results table needs the bot's score against weak engines. "Stockfish level N" is not a
standard: it can mean the UCI `Skill Level` option, Lichess's AI levels, an `UCI_Elo` setting,
or a fixed search depth. Each one gives a different strength.

## Options
1. **UCI Skill Level 0/1/2** with a fixed per-move limit.
2. **Lichess AI levels 1-3**: copy Lichess's per-level settings.
3. **UCI_Elo ladder**: the minimum is ~1320 Elo, likely too strong for this model.
4. **Fixed depth 1/2/3**: deterministic, so repeated games are identical, and depth 1 is
   already a sharp tactician.

## Decision
Option 1. Level 1 = `Skill Level 0`, level 2 = `Skill Level 1`, level 3 = `Skill Level 2`.
Stockfish 19, `Threads 1`, `Hash 16`, search limit 50,000 nodes per move. 50 games per level,
alternating colours, so 150 games in all. The model's first try is greedy (temperature 0);
after an illegal output it gets up to 5 retries **sampled at temperature 1** (a greedy retry
would write the same illegal string again). No legal move after the retries is a forfeit (a
loss, reported separately). Games end on checkmate, stalemate or insufficient material, on
threefold repetition or the 50-move rule once they have actually happened (ended
automatically, like Lichess and `web/js/game.js`), or are adjudicated a draw at 400 plies.

## Why
- Skill Level is Stockfish's own weakening: it searches, then picks among its top moves with a
  level-dependent random error. Games vary from one to the next even with a greedy model.
- A node limit, not a time limit, keeps strength the same on the laptop and on Kaggle.
- Greedy play measures the model's best guess. Sampling every move would add noise that 50
  games can't average out; sampling only the retries keeps that noise to the rare positions
  where the best guess is illegal.

## Revision (2026-11-10, same day)
The first version said "greedy with 5 retries", which can't work: retries must be sampled. The
draw rule was also made explicit. python-chess's `outcome(claim_draw=True)` ends a game when
the side to move *could* claim threefold by announcing its next move, one ply before the
repetition actually happens. `eval/vs_stockfish.py` uses `is_repetition(3)` /
`is_fifty_moves()` instead (a test pins this).

## Consequences
- Results are "score vs Stockfish 19 Skill Level k at 50k nodes". Say that in the README;
  don't call them Elo.
- Forfeits are reported next to the score, so an illegal-move problem can't hide in a loss.
- 50 games per level gives about a ±13-point 95% interval on the score.
