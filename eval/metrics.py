"""Evaluation metrics.

Design: *generation* (running the model, expensive) is separate from *metrics* (pure
functions over records, cheap and easy to test). One pass of evaluate_positions feeds the
legal-move rate, the per-move-number breakdown and the human-move match rate.

Definitions, and why:
- Positions: every position of held-out games (val split), teacher-forced: the prompt holds
  the real move history and the game's real result as prefix, exactly as in training, so the
  metric measures the model, not a distribution shift in the prompt.
- Legal-move rate: fraction of positions where the FIRST greedy output (temperature 0) is
  exactly the SAN of a legal move (strict: python-chess's lenient parse_san would accept
  phantom captures/checks and "--"; exact match is also what the JS port does). Strict (no retries) and deterministic (no sampling noise),
  so two checkpoints can be compared on identical positions.
- Legal rate by move number: same, bucketed by full-move number (1-10, 11-20, ...), keyed by
  the bucket's first move. Long games are rare, so late buckets can be filtered by min_count.
- Move match rate (extra): fraction of positions where the output is exactly the move the human
  played. Not "correctness" (humans blunder), but a useful imitation measure.
- Puzzle solved: Lichess rule. The puzzle's first move is the opponent's; the model must then
  play every solution move, greedy, one try each, with the opponent's replies forced from the
  solution. Any checkmating move also counts (Lichess accepts alternative mates).
  The model is conditioned on winning: prefix "1-0" if it plays White, "0-1" if Black.
- Score vs engine: from the model's side, win 1 / draw 0.5 / loss 0. Games the model forfeits
  (no legal move within its retries) count as losses AND are reported separately, so a weak
  score can be told apart from an illegal-move problem.

Questions to be able to answer:
  - Why teacher forcing with real histories rather than self-play for the legal-move rate?
  - Why greedy decoding for metrics, when the bot samples at temperature > 0 when playing?
  - Why must puzzle evaluation rebuild the full game history instead of starting from a FEN?
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import chess
import torch

from chessgpt.movetext import RESULTS, parse_movetext
from chessgpt.player import build_prompt, legal_move_from_text, replay, sample_move_string
from chessgpt.tokenizer import CharTokenizer


@dataclass
class PositionRecord:
    game: int            # index into the movetexts given to evaluate_positions
    ply: int             # 0 = White's first move
    move_number: int     # full-move number, ply // 2 + 1
    output: str          # what the model wrote (may be anything)
    legal: bool          # output parses as a legal move in this position
    matches_game: bool   # output is the move the human actually played


@dataclass
class Puzzle:
    id: str
    rating: int
    history: list[str]   # SAN moves from the start position up to the puzzle position
    moves: list[str]     # UCI: moves[0] = opponent's setup move, then model / opponent alternately


def _boundary_ids(tokenizer: CharTokenizer, delimiter: str) -> set[int]:
    return {tokenizer.stoi[c] for c in (" ", delimiter) if c in tokenizer.stoi}


def _parse(board: chess.Board, text: str) -> chess.Move | None:
    # Strict: exact SAN of a legal move (see chessgpt.player.legal_move_from_text).
    return legal_move_from_text(board, text)


# --- generation --------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_positions(
    model, tokenizer: CharTokenizer, movetexts: Iterable[str], *, temperature: float = 0.0,
    max_plies: int | None = None, delimiter: str = "\n", generator: torch.Generator | None = None,
) -> list[PositionRecord]:
    """Ask the model for a move in every position of every game (teacher-forced).

    movetexts: games in the canonical format, e.g. "1-0 1.e4 e5 2.Nf3" (val games from games.jsonl).
    max_plies: evaluate at most this many positions per game (None = all).
    """
    model.eval()
    boundary = _boundary_ids(tokenizer, delimiter)
    records = []
    for g, movetext in enumerate(movetexts):
        result, sans = parse_movetext(movetext)
        board = chess.Board()
        for ply, actual in enumerate(sans[:max_plies] if max_plies is not None else sans):
            prompt = tokenizer.encode(build_prompt(sans[:ply], result, delimiter))
            output = sample_move_string(model, tokenizer, prompt, temperature, boundary, generator)
            move = _parse(board, output)
            records.append(PositionRecord(
                game=g, ply=ply, move_number=ply // 2 + 1, output=output,
                legal=move is not None,
                matches_game=move is not None and move == board.parse_san(actual),
            ))
            board.push_san(actual)
    return records


@torch.no_grad()
def solve_puzzle(model, tokenizer: CharTokenizer, puzzle: Puzzle, *, delimiter: str = "\n") -> bool:
    """True if the model plays every solution move (or any mate), greedy, one try each."""
    model.eval()
    boundary = _boundary_ids(tokenizer, delimiter)
    board = replay(puzzle.history)
    sans = list(puzzle.history)

    setup = chess.Move.from_uci(puzzle.moves[0])
    sans.append(board.san(setup))
    board.push(setup)
    prefix = "1-0" if board.turn == chess.WHITE else "0-1"  # the model's side wins

    for i in range(1, len(puzzle.moves)):
        expected = chess.Move.from_uci(puzzle.moves[i])
        if i % 2 == 1:  # model's move
            prompt = tokenizer.encode(build_prompt(sans, prefix, delimiter))
            move = _parse(board, sample_move_string(model, tokenizer, prompt, 0.0, boundary))
            if move is None:
                return False
            if move != expected:
                board.push(move)
                return board.is_checkmate()  # alternative mate accepted, anything else fails
        sans.append(board.san(expected))
        board.push(expected)
    return True


# --- metrics over records ----------------------------------------------------------------------
def legal_move_rate(records: list[PositionRecord]) -> float:
    if not records:
        raise ValueError("no positions evaluated")
    return sum(r.legal for r in records) / len(records)


def move_match_rate(records: list[PositionRecord]) -> float:
    if not records:
        raise ValueError("no positions evaluated")
    return sum(r.matches_game for r in records) / len(records)


def legal_rate_by_move_number(
    records: list[PositionRecord], bucket_size: int = 10, min_count: int = 1,
) -> dict[int, float]:
    """{first move number of bucket: legal rate}, e.g. {1: .., 11: .., 21: ..} for size 10.
    Buckets with fewer than min_count positions are left out."""
    buckets: dict[int, list[bool]] = defaultdict(list)
    for r in records:
        start = (r.move_number - 1) // bucket_size * bucket_size + 1
        buckets[start].append(r.legal)
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items()) if len(v) >= min_count}


def puzzle_accuracy(
    results: list[tuple[int, bool]], edges: tuple[int, ...] = (1000, 1500, 2000, 2500),
) -> dict[str, float]:
    """results: (puzzle rating, solved). Returns accuracy per rating bucket (non-empty ones)
    plus "all". With the default edges: "<1000", "1000-1499", ..., "2500+"."""
    if not results:
        raise ValueError("no puzzles evaluated")
    labels = [f"<{edges[0]}"] + [f"{a}-{b - 1}" for a, b in zip(edges, edges[1:])] + [f"{edges[-1]}+"]
    buckets: dict[str, list[bool]] = defaultdict(list)
    for rating, solved in results:
        idx = sum(rating >= e for e in edges)
        buckets[labels[idx]].append(solved)
    out = {label: sum(buckets[label]) / len(buckets[label]) for label in labels if buckets[label]}
    out["all"] = sum(s for _, s in results) / len(results)
    return out


def score_vs_engine(
    results: list[str], model_colors: list[str], forfeits: list[bool] | None = None,
) -> dict:
    """Score from the model's side. results: "1-0" / "0-1" / "1/2-1/2" per game;
    model_colors: "white" / "black"; forfeits: True where the model failed to produce a
    legal move (counted as a loss whatever the result string says)."""
    forfeits = forfeits if forfeits is not None else [False] * len(results)
    if not (len(results) == len(model_colors) == len(forfeits)):
        raise ValueError("results, model_colors and forfeits must have the same length")
    wins = draws = losses = 0
    for result, color, forfeit in zip(results, model_colors, forfeits):
        if result not in RESULTS:
            raise ValueError(f"unknown result {result!r} (expected one of {RESULTS})")
        if color not in ("white", "black"):
            raise ValueError(f"model color must be 'white' or 'black', got {color!r}")
        if forfeit:
            losses += 1
        elif result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == (color == "white"):
            wins += 1
        else:
            losses += 1
    n = len(results)
    return {"games": n, "wins": wins, "draws": draws, "losses": losses,
            "forfeits": sum(forfeits), "score": (wins + 0.5 * draws) / n if n else 0.0}
