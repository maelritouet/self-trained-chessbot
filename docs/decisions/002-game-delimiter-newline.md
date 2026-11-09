# 002: Separate games with a newline

- **Date:** 2026-11-09
- **Status:** accepted

## Context
`train.bin` is one long stream of characters with all games concatenated (nanoGPT style).
A delimiter character marks where a game starts, so the model (and the player's prompt)
know a new game begins.

## Options
1. **`;`** as in Karvonen's Chess-GPT. Never appears in SAN; makes results directly comparable
   with that work's format.
2. **Newline `\n`**. Never appears in SAN either; the decoded `.bin` reads as one game per line,
   matching `games.jsonl`.

Functionally the two are identical: one reserved character that only ever means "new game".

## Decision
Newline (`delimiter: "\n"` in `configs/data.yaml`).

## Why
The choice has no effect on what the model can learn, so readability wins: decoded training
data and prompts read as one game per line, which makes debugging and inspecting samples easier.

## Consequences
- Each game is written as `\n<result> 1.e4 …`: the delimiter comes *before* the game, so the model
  always sees "new game" right before the result and first move. The player's prompt starts with `\n`.
- The delimiter is stored in `meta.json` and in `best.pt`, so the player and the web demo read it
  rather than hard-coding it.
- When comparing against Karvonen's models, remember their delimiter is `;`.
