# 008: A model output is a legal move only if it is the exact SAN of a legal move

- **Date:** 2026-11-09 (applied in `f3fdc76`), logged 2026-11-10
- **Status:** accepted

## Context
The model writes characters; something must decide whether they are a move. At first,
`choose_move` and the metrics used python-chess's `Board.parse_san`, and the web page used
chess.js `move()`. Both are lenient. They accept a phantom capture or check (`Nxf3` or `Nf3+`
for a quiet `Nf3`), long algebraic (`g1f3`), and python-chess accepts `--` as a **null move**,
so the bot could have "played" a pass. The two libraries are also lenient in different ways,
so the Python and JS players could disagree on the same output.

## Options
1. **Lenient:** accept anything the library parses into a legal move.
2. **Exact SAN:** the output must equal `board.san(m)` for some legal move `m`
   (`chessgpt.player.legal_move_from_text`, `legalSan` in `web/js/player.js`).
3. **Normalised:** strip or repair `x`/`+`/`#` before matching, and reject the rest.

## Decision
Exact SAN (option 2), everywhere: the player, the metrics and the web page.

## Why
- It is the only definition with no exceptions to explain. The training data is exact SAN, so
  "the model learned the notation" and "the model wrote a legal move" become the same claim.
- Python and JS agree by construction: both compare against a list of generated SAN strings,
  not against two different parsers.
- It closes the null-move hole, which no repair rule (option 3) would cover for every case
  without a list of special cases.

## Consequences
- The legal-move rate is stricter than it was. On the base model (2026-11-10, 20 val games,
  1,784 positions): **95.1%** strict. 23 of the 88 illegal outputs would pass a lenient parse
  (wrong or missing `x`, `+` or disambiguation on a real move), so lenient would give about
  96.4%. The session-1 figure of 81% (test model) was measured leniently and is not comparable.
- Errors split into "wrong chess" (unreachable or illegal move, ignored check: 55 of 88) and
  "wrong notation" (33 of 88: 23 lenient-parsable, 10 missing disambiguation). Report both
  when discussing the metric.
- Project invariant: never call `parse_san` or chess.js `move()` on model text.
