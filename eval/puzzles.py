"""Lichess puzzle evaluation (decision 010).

    # 1. Sample puzzles, fetch their source games from Lichess, write the puzzle set (once):
    python -m eval.puzzles prepare data/raw/puzzles/lichess_db_puzzle.csv.zst --out data/puzzles/puzzles.jsonl
    # 2. Score a checkpoint on that fixed set:
    python -m eval.puzzles run runs/<run>/best.pt --puzzles data/puzzles/puzzles.jsonl --out results/puzzles.jsonl

Our model reads movetext from move 1, not a FEN, so each puzzle needs the real game history.
Each CSV row has a GameUrl; the source games are fetched with the Lichess export API (up to
300 ids per request, ndjson). The history is the game prefix whose position matches the
puzzle's FEN (pieces, side to move, castling rights; not clocks or en passant notation). A
puzzle whose game can't be fetched or never reaches the FEN is dropped and counted by reason.

Sampling: stratified by puzzle rating (the buckets of eval.metrics.puzzle_accuracy), only
settled ratings (RatingDeviation <= max_rd), one seeded reservoir per bucket, so a single
streaming pass over the ~5M-row CSV gives a uniform sample per bucket.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import chess

from eval.metrics import Puzzle

# post(url, data, headers) -> (status_code, body_text)
Post = Callable[[str, str, dict], tuple[int, str]]

EXPORT_URL = "https://lichess.org/api/games/export/_ids?moves=true&tags=false&clocks=false&evals=false"
RATE_LIMIT_WAIT_S = 61  # Lichess API guidance: on HTTP 429, wait a full minute


def parse_game_url(url: str) -> str:
    """'https://lichess.org/game/787zsVup/black#48' (or without 'game/') -> '787zsVup'."""
    parts = url.split("#", 1)[0].split("lichess.org/", 1)[1].split("/")
    return parts[1] if parts[0] == "game" else parts[0]


def _key(board: chess.Board) -> tuple:
    return board.board_fen(), board.turn, board.castling_rights


def history_to_position(game_sans: list[str], fen: str) -> list[str] | None:
    """Shortest prefix of the game whose position is the FEN's, or None if never reached.
    (Never the game's final position: a puzzle's moves are later moves of the same game.)"""
    target = _key(chess.Board(fen))
    board = chess.Board()
    for i, san in enumerate(game_sans):
        if _key(board) == target:
            return game_sans[:i]
        board.push_san(san)
    return None


def select_puzzles(rows: Iterable[dict], per_bucket: int, edges: tuple[int, ...] = (1000, 1500, 2000, 2500),
                   max_rd: int = 90, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    reservoirs: list[list[dict]] = [[] for _ in range(len(edges) + 1)]
    seen = [0] * (len(edges) + 1)
    for r in rows:
        if int(r["RatingDeviation"]) > max_rd:
            continue
        b = sum(int(r["Rating"]) >= e for e in edges)
        seen[b] += 1
        if len(reservoirs[b]) < per_bucket:
            reservoirs[b].append(r)
        else:  # Algorithm R: keep the new row with probability per_bucket / seen
            j = rng.randrange(seen[b])
            if j < per_bucket:
                reservoirs[b][j] = r
    return [r for res in reservoirs for r in res]


def fetch_games(ids: list[str], post: Post, batch_size: int = 300,
                sleep: Callable[[float], None] = time.sleep) -> dict[str, list[str]]:
    """{game id: SAN moves} for the standard-chess games among `ids`."""
    games: dict[str, list[str]] = {}
    headers = {"Accept": "application/x-ndjson", "Content-Type": "text/plain"}
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        while True:
            status, body = post(EXPORT_URL, ",".join(batch), headers)
            if status != 429:
                break
            sleep(RATE_LIMIT_WAIT_S)
        if status != 200:
            raise RuntimeError(f"Lichess export failed: HTTP {status}: {body[:200]}")
        for line in body.splitlines():
            if line.strip():
                g = json.loads(line)
                if g.get("variant", "standard") == "standard" and "initialFen" not in g:
                    games[g["id"]] = g["moves"].split()
    return games


def build_puzzles(rows: list[dict], games: dict[str, list[str]]) -> tuple[list[Puzzle], Counter]:
    puzzles, dropped = [], Counter()
    for r in rows:
        sans = games.get(parse_game_url(r["GameUrl"]))
        if sans is None:
            dropped["game_not_fetched"] += 1
            continue
        history = history_to_position(sans, r["FEN"])
        if history is None:
            dropped["fen_not_reached"] += 1
            continue
        puzzles.append(Puzzle(id=r["PuzzleId"], rating=int(r["Rating"]), history=history, moves=r["Moves"].split()))
    return puzzles, dropped


# --- CLI ---------------------------------------------------------------------------------------
def _urllib_post(url: str, data: str, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, data=data.encode(), headers={**headers, "User-Agent": "chessgpt-eval"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def _csv_rows(path: str):
    import zstandard
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        yield from csv.DictReader(text)


def _prepare(a) -> None:
    rows = select_puzzles(_csv_rows(a.csv), a.per_bucket, max_rd=a.max_rd, seed=a.seed)
    print(f"sampled {len(rows)} puzzles", flush=True)
    games = fetch_games(sorted({parse_game_url(r["GameUrl"]) for r in rows}), _urllib_post)
    puzzles, dropped = build_puzzles(rows, games)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in puzzles:
            f.write(json.dumps(asdict(p)) + "\n")
    stats = {"sampled": len(rows), "kept": len(puzzles), "dropped": dict(dropped),
             "per_bucket": a.per_bucket, "max_rd": a.max_rd, "seed": a.seed, "source": a.csv}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))


def _run(a) -> None:
    import torch

    from chessgpt.model import GPT, GPTConfig
    from chessgpt.tokenizer import CharTokenizer
    from eval.metrics import puzzle_accuracy, solve_puzzle

    ck = torch.load(a.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ck["model_args"]))
    model.load_state_dict(ck["model"])
    model.to(a.device).eval()
    tok = CharTokenizer(ck["itos"])
    delim = ck.get("delimiter", "\n")
    puzzles = [Puzzle(**json.loads(line)) for line in open(a.puzzles)]
    results = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, p in enumerate(puzzles):
            solved = solve_puzzle(model, tok, p, delimiter=delim)
            results.append((p.rating, solved))
            f.write(json.dumps({"ckpt": a.ckpt, "id": p.id, "rating": p.rating, "solved": solved}) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(puzzles)}: {puzzle_accuracy(results)}", flush=True)
    print(json.dumps({"ckpt": a.ckpt, "puzzles": len(results), "accuracy": puzzle_accuracy(results)}))


def main(argv=None) -> None:
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare", help="sample puzzles and fetch their games (network)")
    pr.add_argument("csv", help="lichess_db_puzzle.csv.zst")
    pr.add_argument("--out", default="data/puzzles/puzzles.jsonl")
    pr.add_argument("--per-bucket", type=int, default=200)
    pr.add_argument("--max-rd", type=int, default=90)
    pr.add_argument("--seed", type=int, default=0)
    ru = sub.add_parser("run", help="score a checkpoint on a prepared puzzle set")
    ru.add_argument("ckpt")
    ru.add_argument("--puzzles", default="data/puzzles/puzzles.jsonl")
    ru.add_argument("--out", required=True, help="JSONL, one line per puzzle")
    ru.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    _prepare(a) if a.cmd == "prepare" else _run(a)


if __name__ == "__main__":
    main()
