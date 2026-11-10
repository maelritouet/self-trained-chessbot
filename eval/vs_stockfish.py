"""Play the model against Stockfish (decision 011).

    python -m eval.vs_stockfish runs/<run>/best.pt --games 50 --levels 0 1 2 --out results/vs_sf.jsonl

The game loop knows nothing about models or engines: it calls two "players", each a function
(board, sans_so_far) -> (move or None, illegal_tries). That keeps the rules testable with
scripted players:
  - Games end on checkmate, stalemate, insufficient material, threefold repetition or the
    50-move rule (draws claimed automatically, like Lichess), or are adjudicated a draw at
    max_plies.
  - A player that returns None forfeits: a loss, recorded in forfeit_by.

The model player tries its greedy move first (the model's best guess), then up to
max_retries samples at temperature 1. Retrying greedily would be pointless: a greedy model
writes the same illegal string every time. It does not call choose_move, which also scores
every legal move (one extra batched forward per move) for the web page's top-5 bars.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import chess
import chess.engine
import torch

from chessgpt.player import build_prompt, legal_move_from_text, sample_move_string

# A player gets the current board and the SAN moves so far, and returns
# (move, illegal_tries): move None means it could not produce a legal move (forfeit).
Player = Callable[[chess.Board, list[str]], tuple[chess.Move | None, int]]

STOCKFISH_PATHS = ("/run/host/usr/bin/stockfish", "/usr/games/stockfish", "/usr/bin/stockfish")


@dataclass
class GameRecord:
    result: str                 # "1-0" / "0-1" / "1/2-1/2"
    termination: str            # "checkmate", "stalemate", "insufficient_material",
                                # "threefold_repetition", "fifty_moves", "max_plies", "forfeit", ...
    sans: list[str] = field(default_factory=list)
    illegal_tries: dict[str, int] = field(default_factory=lambda: {"white": 0, "black": 0})
    forfeit_by: str | None = None   # "white" / "black" if that side forfeited


def play_game(white: Player, black: Player, max_plies: int = 400) -> GameRecord:
    board = chess.Board()
    rec = GameRecord(result="*", termination="")
    sides = {chess.WHITE: (white, "white"), chess.BLACK: (black, "black")}
    while True:
        outcome = board.outcome()  # automatic endings (incl. fivefold / 75 moves)
        if outcome is not None:
            rec.result, rec.termination = outcome.result(), outcome.termination.name.lower()
            return rec
        # Claimable draws, ended like Lichess once they have actually happened. Not
        # outcome(claim_draw=True): that also ends the game when the side to move *could* claim
        # by announcing a repeating move it has not played yet.
        if board.is_repetition(3) or board.is_fifty_moves():
            rec.result = "1/2-1/2"
            rec.termination = "threefold_repetition" if board.is_repetition(3) else "fifty_moves"
            return rec
        if len(rec.sans) >= max_plies:
            rec.result, rec.termination = "1/2-1/2", "max_plies"
            return rec
        player, side = sides[board.turn]
        move, illegal = player(board.copy(), list(rec.sans))
        rec.illegal_tries[side] += illegal
        if move is None:
            rec.result = "0-1" if side == "white" else "1-0"
            rec.termination, rec.forfeit_by = "forfeit", side
            return rec
        rec.sans.append(board.san(move))
        board.push(move)


def model_player(model, tokenizer, color: str, *, max_retries: int = 5, retry_temperature: float = 1.0,
                 delimiter: str = "\n", generator=None) -> Player:
    """The model as a player, conditioned on its own side winning ("1-0" as white, "0-1" as black)."""
    prefix = "1-0" if color == "white" else "0-1"
    boundary = {tokenizer.stoi[c] for c in (" ", delimiter) if c in tokenizer.stoi}

    @torch.no_grad()
    def play(board: chess.Board, sans: list[str]):
        model.eval()
        prompt = tokenizer.encode(build_prompt(sans, prefix, delimiter))
        for attempt in range(1 + max_retries):
            temperature = 0.0 if attempt == 0 else retry_temperature
            move = legal_move_from_text(
                board, sample_move_string(model, tokenizer, prompt, temperature, boundary, generator))
            if move is not None:
                return move, attempt
        return None, 1 + max_retries
    return play


def stockfish_player(engine: chess.engine.SimpleEngine, skill: int, nodes: int) -> Player:
    def play(board: chess.Board, sans: list[str]):
        engine.configure({"Skill Level": skill})
        return engine.play(board, chess.engine.Limit(nodes=nodes)).move, 0
    return play


def find_stockfish() -> str | None:
    return shutil.which("stockfish") or next((p for p in STOCKFISH_PATHS if Path(p).exists()), None)


# --- CLI ---------------------------------------------------------------------------------------
def main(argv=None) -> None:
    from chessgpt.model import GPT, GPTConfig
    from chessgpt.tokenizer import CharTokenizer
    from eval.metrics import score_vs_engine

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", help="best.pt written by chessgpt.train")
    p.add_argument("--out", required=True, help="JSONL file, one line per game (appended as games finish)")
    p.add_argument("--games", type=int, default=50, help="games per level, colours alternate")
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2], help="Stockfish 'Skill Level' values")
    p.add_argument("--nodes", type=int, default=50_000, help="Stockfish search limit per move")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--stockfish", default=None, help="path to the binary (default: search PATH and host paths)")
    p.add_argument("--seed", type=int, default=0, help="seed for the model's retry sampling")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    sf = a.stockfish or find_stockfish()
    if sf is None:
        raise SystemExit("no stockfish binary found: pass --stockfish")
    ck = torch.load(a.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ck["model_args"]))
    model.load_state_dict(ck["model"])
    model.to(a.device).eval()
    tok = CharTokenizer(ck["itos"])
    delim = ck.get("delimiter", "\n")
    gen = torch.Generator(device=a.device).manual_seed(a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with chess.engine.SimpleEngine.popen_uci(sf) as engine, out.open("a") as f:
        engine.configure({"Threads": 1, "Hash": 16})
        for level in a.levels:
            results, colors, forfeits = [], [], []
            for g in range(a.games):
                color = "white" if g % 2 == 0 else "black"
                me = model_player(model, tok, color, max_retries=a.max_retries, delimiter=delim, generator=gen)
                sf_player = stockfish_player(engine, level, a.nodes)
                t0 = time.time()
                rec = play_game(*((me, sf_player) if color == "white" else (sf_player, me)), max_plies=a.max_plies)
                row = {"ckpt": a.ckpt, "skill": level, "nodes": a.nodes, "game": g, "model_color": color,
                       "seconds": round(time.time() - t0, 1), **asdict(rec)}
                f.write(json.dumps(row) + "\n")
                f.flush()
                results.append(rec.result)
                colors.append(color)
                forfeits.append(rec.forfeit_by == color)
                print(f"skill {level} game {g + 1}/{a.games}: model {color} {rec.result} "
                      f"({rec.termination}, {len(rec.sans)} plies, {rec.illegal_tries[color]} illegal)", flush=True)
            summary[level] = score_vs_engine(results, colors, forfeits)
            print(f"skill {level}: {summary[level]}", flush=True)
    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main()
