"""Evaluation metric definitions.

Signatures are a starting point and may change. Each docstring lists the choices
the definition has to pin down; write the answers into the docstring (and a
docs/decisions/ entry where it matters) before implementing.
"""

from __future__ import annotations


def legal_move_rate(model, tokenizer, games: list[list[str]], **kwargs) -> float:
    """Fraction of positions where the model's move is legal.

    To decide:
      - positions from real held-out games (teacher-forced history) or from self-play?
      - "legal" = first sample legal, or legal within max_retries?
      - temperature used (0 = greedy is the most reproducible)
      - which result prefix is in the prompt?
    """
    raise NotImplementedError


def legal_rate_by_move_number(model, tokenizer, games: list[list[str]], **kwargs) -> dict[int, float]:
    """Legal-move rate bucketed by move number (or ply). Expect it to fall in long games.

    To decide: bucket size, minimum positions per bucket to report, behaviour past block_size.
    """
    raise NotImplementedError


def puzzle_accuracy(model, tokenizer, puzzles, **kwargs) -> dict[str, float]:
    """Accuracy on Lichess puzzles, bucketed by puzzle rating.

    To decide: solved = every solution move correct, or just the first one?
    Rating buckets? How is the move history before the puzzle position built?
    """
    raise NotImplementedError


def score_vs_engine(results: list[str], model_colors: list[str]) -> dict[str, float]:
    """Score against an engine from game results: win=1, draw=0.5, loss=0.

    To decide: how are games where the model can't produce a legal move scored
    (forfeit? fallback move?), and is that reported separately?
    """
    raise NotImplementedError
