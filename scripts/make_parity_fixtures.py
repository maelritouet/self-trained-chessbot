"""Generate the JS-vs-Python parity fixtures for web/test/player.test.mjs.

    python -m chessgpt.train --config web/test/fixtures/tiny.yaml --data data/processed --out runs/parity-tiny
    python scripts/make_parity_fixtures.py runs/parity-tiny/best.pt

Exports the checkpoint to web/test/fixtures/tiny/ (model.onnx + meta.json, small enough to
commit) and writes web/test/fixtures/cases.json: for a set of positions, what the Python
choose_move returns at temperature 0 (greedy, deterministic): the move, the illegal tries,
and the probability of EVERY legal move. The JS port must reproduce all of it, which checks
the prompt format, tokenization, sampling loop, scoring maths, and that chess.js generates
the same legal moves with the same SAN as python-chess.

Positions come from held-out games (val.bin) at several plies, plus a position just before
a promotion and one whose prompt is longer than block_size (tests cropping).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from chessgpt.model import GPT, GPTConfig
from chessgpt.movetext import parse_movetext
from chessgpt.player import build_prompt, choose_move
from chessgpt.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "web" / "test" / "fixtures"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_onnx", ROOT / "scripts" / "export_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_positions(val_games: list[str], block_size: int) -> list[dict]:
    positions = [
        {"moves": [], "result_prefix": "1-0"},
        {"moves": ["e4"], "result_prefix": "0-1"},
        {"moves": ["e4", "e5"], "result_prefix": None},  # no prefix: out of distribution, still must match
    ]
    for movetext in val_games[:4]:
        result, sans = parse_movetext(movetext)
        for ply in (10, 30, 60):
            if ply < len(sans):
                positions.append({"moves": sans[:ply], "result_prefix": result})
    # Just before a promotion (SAN with "=").
    for movetext in val_games:
        result, sans = parse_movetext(movetext)
        k = next((i for i, s in enumerate(sans) if "=" in s), None)
        if k is not None:
            positions.append({"moves": sans[:k], "result_prefix": result})
            break
    # A prompt longer than block_size (exercises left-cropping).
    for movetext in val_games:
        result, sans = parse_movetext(movetext)
        if len(build_prompt(sans[:-1], result)) > block_size + 50:
            positions.append({"moves": sans[: len(sans) - 1], "result_prefix": result})
            break
    return positions


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--val", default="data/processed/val.bin")
    a = p.parse_args(argv)

    out = _load_exporter().export_onnx(a.ckpt, FIXTURES / "tiny")
    ck = torch.load(a.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ck["model_args"]))
    model.load_state_dict(ck["model"])
    model.eval()
    tok = CharTokenizer(ck["itos"])
    delimiter = ck.get("delimiter", "\n")

    val_text = tok.decode(np.fromfile(a.val, dtype=np.uint8))
    val_games = [g for g in val_text.split(delimiter) if g]
    cases = []
    for pos in pick_positions(val_games, ck["model_args"]["block_size"]):
        c = choose_move(model, tok, pos["moves"], temperature=0.0, max_retries=2,
                        result_prefix=pos["result_prefix"], top_k=10_000, delimiter=delimiter)
        cases.append({
            **pos,
            "san": c.san,
            "illegal_tries": c.illegal_tries,
            "candidates": [[x.san, x.prob] for x in c.candidates],
            "legal_mass": c.legal_mass,
        })
    (FIXTURES / "cases.json").write_text(json.dumps({"max_retries": 2, "cases": cases}, indent=1) + "\n")
    legal = sum(c["san"] is not None for c in cases)
    print(f"{out}: {len(cases)} cases ({legal} with a legal greedy move), "
          f"{(out / 'model.onnx').stat().st_size / 1e3:.0f} kB onnx")


if __name__ == "__main__":
    main()
