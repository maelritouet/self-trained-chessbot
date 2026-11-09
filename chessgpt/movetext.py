"""The canonical movetext format, shared by the data pipeline, the player and evals.

A game is a single line:  [<result> ]1.e4 e5 2.Nf3 Nc6 3.Bb5 ...
- moves are SAN, white moves prefixed with "<n>." (no space after the dot)
- tokens are separated by single spaces
- the optional result ("1-0", "0-1", "1/2-1/2") comes first (configs/data.yaml:
  result_position: prefix), or last with "suffix".

Keeping this in one module means the downloader that writes the format and the
player that builds prompts in it can't drift apart.
"""

from __future__ import annotations

RESULTS = ("1-0", "0-1", "1/2-1/2")


def format_movetext(sans: list[str]) -> str:
    """['e4', 'e5', 'Nf3'] -> '1.e4 e5 2.Nf3'."""
    parts = []
    for i, san in enumerate(sans):
        parts.append(f"{i // 2 + 1}.{san}" if i % 2 == 0 else san)
    return " ".join(parts)


def parse_movetext(movetext: str) -> tuple[str | None, list[str]]:
    """'1-0 1.e4 e5 2.Nf3' -> ('1-0', ['e4', 'e5', 'Nf3']).

    Accepts the result as prefix or suffix (or absent -> None). Does not check
    legality; that's python-chess's job.
    """
    result, sans = None, []
    for tok in movetext.split():
        if tok in RESULTS:
            result = tok
            continue
        _, dot, san = tok.partition(".")
        sans.append(san if dot else tok)
    return result, sans
