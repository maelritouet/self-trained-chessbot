"""Stream a Lichess monthly dump, filter and normalise games, write games.jsonl.

The monthly files on https://database.lichess.org are ~30 GB of zstd-compressed
PGN, so nothing is saved to disk except the games we keep: the HTTP response is
decompressed on the fly and read line by line, and the connection is closed once
`max_games` games have passed the filters.

Output: one JSON object per line in data/processed/games.jsonl:
    {"movetext": "1-0 1.e4 e5 2.Nf3 Nc6 ...", "white_elo": 1834, "black_elo": 1790,
     "result": "1-0", "speed": "blitz"}

Usage:
    python scripts/download_lichess.py                       # uses configs/data.yaml
    python scripts/download_lichess.py --max-games 20000     # quick run
    python scripts/download_lichess.py --month 2026-09
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml
import zstandard
from tqdm import tqdm

from chessgpt.movetext import format_movetext

LIST_URL = "https://database.lichess.org/standard/list.txt"
MONTH_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"

RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}

# --- Movetext normalisation -------------------------------------------------
# Raw Lichess movetext looks like (with clocks/evals on some games):
#   1. e4 { [%eval 0.2] [%clk 0:03:00] } 1... e5?! { [%clk 0:03:00] } 2. Nf3 $1 ... 1-0
COMMENT_RE = re.compile(r"\{[^}]*\}")      # { ... } comments, incl. [%clk] / [%eval]
VARIATION_RE = re.compile(r"\([^()]*\)")   # ( ... ) variations (not in Lichess dumps, cheap to handle)
NAG_RE = re.compile(r"\$\d+")              # numeric annotation glyphs: $1, $14, ...
MOVE_NUMBER_RE = re.compile(r"^\d+\.(\.\.)?$")  # "12." or "12..." as a separate token
GLUED_NUMBER_RE = re.compile(r"^\d+\.+")   # "12.e4" / "12...e5" glued to the move
ANNOTATION_CHARS = "!?"                    # suffix annotations: e4!, e4?!, e4??


def extract_sans(raw_movetext: str) -> list[str]:
    """Return the list of SAN moves in a raw PGN movetext, dropping everything else.

    Strips comments, variations, NAGs, move numbers (white "1." and black "1..."),
    !/? annotations and the trailing result token.
    """
    text = COMMENT_RE.sub(" ", raw_movetext)
    while "(" in text:  # nested variations: strip innermost first
        stripped = VARIATION_RE.sub(" ", text)
        if stripped == text:
            break
        text = stripped
    text = NAG_RE.sub(" ", text)

    sans = []
    for tok in text.split():
        if tok in RESULTS or MOVE_NUMBER_RE.match(tok):
            continue
        tok = GLUED_NUMBER_RE.sub("", tok).rstrip(ANNOTATION_CHARS)
        if tok:
            sans.append(tok)
    return sans


def normalise_movetext(
    raw_movetext: str, result: str | None = None, result_position: str = "none"
) -> tuple[str, int]:
    """Normalise raw movetext. Returns (movetext, number_of_plies).

    result_position: "prefix" -> '1-0 1.e4 e5 ...', "suffix" -> '1.e4 e5 ... 1-0',
    "none" -> '1.e4 e5 ...'.
    """
    if result_position not in ("prefix", "suffix", "none"):
        raise ValueError(f"result_position must be prefix/suffix/none, got {result_position!r}")
    sans = extract_sans(raw_movetext)
    parts = [format_movetext(sans)]
    if result is not None and result_position == "prefix":
        parts.insert(0, result)
    elif result is not None and result_position == "suffix":
        parts.append(result)
    return " ".join(p for p in parts if p), len(sans)


# --- Filters ------------------------------------------------------------------
EVENT_RE = re.compile(r"^Rated (\w+) ")  # "Rated Blitz game", "Rated Rapid tournament https://..."


def event_speed(event: str) -> str | None:
    """'Rated Blitz game' -> 'blitz'. None if the event isn't a rated game."""
    m = EVENT_RE.match(event + " ")
    return m.group(1).lower() if m else None


def parse_elo(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:  # "?" for unrated/provisional accounts
        return None


@dataclass
class Stats:
    scanned: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)
    duplicates: int = 0
    total_chars: int = 0
    total_plies: int = 0
    speeds: Counter = field(default_factory=Counter)
    charset: set = field(default_factory=set)


def filter_and_normalise(headers: dict[str, str], raw_movetext: str, cfg: dict, stats: Stats):
    """Apply the filters in order; return a record dict, or None (and count the reason)."""
    speed = event_speed(headers.get("Event", ""))
    if speed not in cfg["speeds"]:
        stats.rejected["speed"] += 1
        return None

    white_elo = parse_elo(headers.get("WhiteElo"))
    black_elo = parse_elo(headers.get("BlackElo"))
    if white_elo is None or black_elo is None or min(white_elo, black_elo) < cfg["min_elo"]:
        stats.rejected["elo"] += 1
        return None

    if headers.get("Termination") in cfg["drop_terminations"]:
        stats.rejected["termination"] += 1
        return None

    result = headers.get("Result", "*")
    if result not in RESULTS - {"*"}:
        stats.rejected["no_result"] += 1
        return None

    movetext, plies = normalise_movetext(raw_movetext, result, cfg["result_position"])
    if plies < cfg["min_plies"]:
        stats.rejected["min_plies"] += 1
        return None

    return {
        "movetext": movetext,
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "speed": speed,
        "_plies": plies,
    }


# --- PGN streaming ------------------------------------------------------------
def iter_games(lines):
    """Yield (headers, raw_movetext) from an iterator of PGN text lines.

    Parsed as plain text (not chess.pgn.read_game, which is ~100x slower):
    a game is a block of '[Key "Value"]' header lines, then movetext lines.
    A header line after movetext marks the start of the next game.
    """
    headers: dict[str, str] = {}
    movetext: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            if movetext:  # new game starts
                yield headers, " ".join(movetext)
                headers, movetext = {}, []
            key, _, value = line[1:-1].partition(" ")
            headers[key] = value.strip('"')
        else:
            movetext.append(line)
    if headers or movetext:
        yield headers, " ".join(movetext)


def latest_month() -> str:
    """Most recent complete month: list.txt lists published dumps, newest first."""
    first_url = requests.get(LIST_URL, timeout=30).text.split()[0]
    return re.search(r"(\d{4}-\d{2})\.pgn\.zst$", first_url).group(1)


def stream_pgn_lines(url: str):
    """Yield decoded text lines from a remote .pgn.zst without saving it."""
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        reader = zstandard.ZstdDecompressor().stream_reader(resp.raw)
        yield from io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


# --- Main -------------------------------------------------------------------------
def print_stats(stats: Stats, seconds: float, month: str, out: Path) -> dict:
    avg_chars = stats.total_chars / max(stats.kept, 1)
    avg_plies = stats.total_plies / max(stats.kept, 1)
    summary = {
        "month": month,
        "games_scanned": stats.scanned,
        "games_kept": stats.kept,
        "rejected": dict(stats.rejected),
        "duplicates": stats.duplicates,
        "kept_by_speed": dict(stats.speeds),
        "avg_chars_per_game": round(avg_chars, 1),
        "avg_plies_per_game": round(avg_plies, 1),
        "charset": "".join(sorted(stats.charset)),
        "charset_size": len(stats.charset),
        "seconds": round(seconds, 1),
    }
    print(f"\n=== {month} -> {out}")
    print(f"scanned      {stats.scanned:>12,}")
    print(f"kept         {stats.kept:>12,}  ({stats.kept / max(stats.scanned, 1):.1%})")
    for reason, n in stats.rejected.most_common():
        print(f"  rejected:{reason:<12}{n:>10,}")
    print(f"  duplicates           {stats.duplicates:>10,}")
    print(f"kept by speed  {dict(stats.speeds)}")
    print(f"avg chars/game {avg_chars:.1f}   avg plies/game {avg_plies:.1f}")
    print(f"charset ({len(stats.charset)}): {summary['charset']!r}")
    print(f"time {seconds:.1f}s  ({stats.scanned / max(seconds, 1e-9):,.0f} games scanned/s)")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--max-games", type=int, help="override max_games from the config")
    p.add_argument("--month", help="override month (YYYY-MM or 'latest')")
    p.add_argument("--out", default="data/processed/games.jsonl")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_games is not None:
        cfg["max_games"] = args.max_games
    if args.month:
        cfg["month"] = args.month
    month = latest_month() if cfg["month"] == "latest" else cfg["month"]
    url = MONTH_URL.format(month=month)
    print(f"Streaming {url}\nkeeping up to {cfg['max_games']:,} games", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    stats, seen = Stats(), set()
    t0 = time.time()

    with open(tmp, "w", encoding="utf-8") as f, tqdm(total=cfg["max_games"], unit="game") as bar:
        try:
            for headers, raw in iter_games(stream_pgn_lines(url)):
                stats.scanned += 1
                rec = filter_and_normalise(headers, raw, cfg, stats)
                if rec is None:
                    continue
                # Dedup on the normalised movetext (8-byte digest keeps the set small).
                digest = hashlib.blake2b(rec["movetext"].encode(), digest_size=8).digest()
                if digest in seen:
                    stats.duplicates += 1
                    continue
                seen.add(digest)

                stats.kept += 1
                stats.total_chars += len(rec["movetext"])
                stats.total_plies += rec.pop("_plies")
                stats.speeds[rec["speed"]] += 1
                stats.charset.update(rec["movetext"])
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                bar.update(1)
                if stats.kept >= cfg["max_games"]:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted: keeping what was collected so far.", file=sys.stderr)

    os.replace(tmp, out)  # only replace the previous file once writing is complete
    summary = print_stats(stats, time.time() - t0, month, out)
    out.with_suffix(".stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
