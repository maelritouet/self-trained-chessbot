"""Legal-move rate on held-out games, with a breakdown of the illegal outputs.

    python -m eval.legal_rate runs/<run>/best.pt --games 100 --out runs/<run>/eval/legal_rate.json

Positions, decoding and the definition of "legal" (exact SAN, decision 008) come from
eval.metrics.evaluate_positions. This adds the CLI and a classification of every illegal
output, so a report can separate "wrong chess" from "wrong notation":

  illegal_move   no legal move matches (unreachable square, pinned piece, ...)
  ignores_check  same, while the side to move is in check
  notation       a real legal move, written wrong (phantom x or +, long algebraic)
  ambiguous      a real move missing its disambiguation (Ne2 when two knights can go)
  not_a_move     not move syntax at all ("", "e9"), or "--" (python-chess's null move)

python-chess's lenient parse_san is used here ONLY to label outputs the strict check has
already rejected. It never decides legality.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import chess
import numpy as np


def classify_illegal(board: chess.Board, text: str) -> str:
    try:
        move = board.parse_san(text)
    except chess.AmbiguousMoveError:
        return "ambiguous"
    except chess.IllegalMoveError:
        return "ignores_check" if board.is_check() else "illegal_move"
    except (chess.InvalidMoveError, ValueError):
        return "not_a_move"
    return "not_a_move" if move == chess.Move.null() else "notation"


def load_games(bin_path, tokenizer, delimiter: str = "\n", n: int | None = None) -> list[str]:
    """Games from a uint8 .bin file (build_dataset.py), split on the delimiter, first n."""
    text = tokenizer.decode(np.fromfile(bin_path, dtype=np.uint8))
    games = [g for g in text.split(delimiter) if g]
    return games if n is None else games[:n]


def main(argv=None) -> None:
    import torch

    from chessgpt.model import GPT, GPTConfig
    from chessgpt.movetext import parse_movetext
    from chessgpt.tokenizer import CharTokenizer
    from eval.metrics import evaluate_positions, legal_move_rate, legal_rate_by_move_number, move_match_rate

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--val", default="data/processed/val.bin")
    p.add_argument("--games", type=int, default=100, help="first N val games (same N = same positions)")
    p.add_argument("--out", required=True, help="JSON summary (plus every illegal output)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    ck = torch.load(a.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ck["model_args"]))
    model.load_state_dict(ck["model"])
    model.to(a.device).eval()
    tok = CharTokenizer(ck["itos"])
    delim = ck.get("delimiter", "\n")
    games = load_games(a.val, tok, delim, a.games)

    t0, records, illegal = time.time(), [], []
    for g, movetext in enumerate(games):
        recs = evaluate_positions(model, tok, [movetext], delimiter=delim)
        _, sans = parse_movetext(movetext)
        board = chess.Board()
        for r in recs:
            r.game = g
            if not r.legal:
                illegal.append({"game": g, "ply": r.ply, "output": r.output, "actual": sans[r.ply],
                                "kind": classify_illegal(board, r.output)})
            board.push_san(sans[r.ply])
        records += recs
        print(f"game {g + 1}/{len(games)}: running legal {legal_move_rate(records):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    summary = {
        "ckpt": a.ckpt, "val": a.val, "games": len(games), "positions": len(records),
        "legal_rate": legal_move_rate(records), "match_rate": move_match_rate(records),
        "legal_by_move": legal_rate_by_move_number(records),
        "positions_by_move": dict(sorted(Counter((r.move_number - 1) // 10 * 10 + 1 for r in records).items())),
        "illegal_kinds": dict(Counter(i["kind"] for i in illegal).most_common()),
        "seconds": round(time.time() - t0, 1),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**summary, "illegal": illegal}, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
