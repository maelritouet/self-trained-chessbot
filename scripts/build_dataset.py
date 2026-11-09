"""Turn games.jsonl into nanoGPT-style train.bin / val.bin + meta.json.

- Split train/val *by game* (shuffle game order with a fixed seed, first
  val_fraction of games -> val), so no game is half in train and half in val.
- Each game is written as  <delimiter><movetext>  so the model always sees the
  delimiter as a "new game starts" token before the first move. At play time the
  prompt is built the same way: "\n1-0 1.e4 e5 ...".
- Tokens are uint8 (vocab is ~35 chars), 1 byte per character.

Usage:
    python scripts/build_dataset.py                    # build from games.jsonl
    python scripts/build_dataset.py --sanity           # build + legality check
    python scripts/build_dataset.py --meta other/meta.json   # reuse an existing vocab
                                                            # (e.g. for a fine-tuning set)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np
import yaml

from chessgpt.movetext import parse_movetext
from chessgpt.tokenizer import CharTokenizer


def read_movetexts(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["movetext"] for line in f]


def write_split(games: list[str], tok: CharTokenizer, delimiter: str, path: Path,
                chunk: int = 100_000) -> int:
    """Encode games to `path` as uint8, in chunks to bound memory. Returns #tokens."""
    n = 0
    with open(path, "wb") as f:
        for i in range(0, len(games), chunk):
            text = "".join(delimiter + g for g in games[i:i + chunk])
            ids = tok.encode_array(text)
            ids.tofile(f)
            n += len(ids)
    return n


def sanity_check(games: list[str], n: int, seed: int) -> bool:
    """Replay n random games with python-chess; every move must be legal SAN."""
    rng = random.Random(seed)
    sample = rng.sample(games, min(n, len(games)))
    failures = []
    for movetext in sample:
        _, sans = parse_movetext(movetext)
        board = chess.Board()
        for ply, san in enumerate(sans):
            try:
                board.push_san(san)
            except ValueError as e:  # IllegalMoveError / InvalidMoveError / AmbiguousMoveError
                failures.append((movetext, ply, san, type(e).__name__))
                break
    ok = len(sample) - len(failures)
    print(f"\nsanity: {ok}/{len(sample)} games replay fully legal ({ok / len(sample):.2%})")
    for movetext, ply, san, err in failures[:5]:
        print(f"  FAIL at ply {ply} ({san!r}, {err}): {movetext[:120]}")
    return not failures


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--games", default="data/processed/games.jsonl")
    p.add_argument("--out", default="data/processed", help="output directory")
    p.add_argument("--meta", help="reuse the vocab from this meta.json instead of building one")
    p.add_argument("--sanity", action="store_true", help="replay 1,000 random games with python-chess")
    p.add_argument("--sanity-n", type=int, default=1000)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    delimiter, val_fraction, seed = cfg["delimiter"], cfg["val_fraction"], cfg["seed"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    games = read_movetexts(Path(args.games))
    if any(delimiter in g for g in games):
        raise ValueError(f"delimiter {delimiter!r} appears inside a game's movetext")
    print(f"read {len(games):,} games from {args.games}")

    # Split by game: shuffle the game order, not the characters.
    order = list(range(len(games)))
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(games) * val_fraction))
    val_games = [games[i] for i in order[:n_val]]
    train_games = [games[i] for i in order[n_val:]]

    if args.meta:
        tok = CharTokenizer.load(args.meta)
    else:
        charset = set().union(*map(set, games))
        tok = CharTokenizer.from_text(charset, extra=delimiter)

    n_train = write_split(train_games, tok, delimiter, out / "train.bin")
    n_val_tok = write_split(val_games, tok, delimiter, out / "val.bin")

    lengths = [len(g) + len(delimiter) for g in games]
    meta_extra = {
        "delimiter": delimiter,
        "dtype": "uint8",
        "n_games": {"train": len(train_games), "val": len(val_games)},
        "n_tokens": {"train": n_train, "val": n_val_tok},
        "avg_game_len": round(sum(lengths) / len(lengths), 1),
        "max_game_len": max(lengths),
        "val_fraction": val_fraction,
        "seed": seed,
        "source": str(args.games),
    }
    tok.save(out / "meta.json", **meta_extra)

    print(f"vocab ({tok.vocab_size}): {''.join(tok.itos)!r}")
    print(f"train: {len(train_games):,} games, {n_train:,} tokens -> {out / 'train.bin'}")
    print(f"val:   {len(val_games):,} games, {n_val_tok:,} tokens -> {out / 'val.bin'}")
    print(f"avg game length {meta_extra['avg_game_len']} chars, max {meta_extra['max_game_len']}")
    print(f"built in {time.time() - t0:.1f}s")

    if args.sanity and not sanity_check(games, args.sanity_n, seed):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
