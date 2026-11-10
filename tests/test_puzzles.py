"""Tests for eval/puzzles.py: building puzzles with their full move history (decision 010).

No network: fetch_games gets a fake `post`. Positions are built with python-chess, so the
FENs are exact.
"""

import json
from collections import Counter

import chess
import pytest

from eval.puzzles import build_puzzles, fetch_games, history_to_position, parse_game_url, select_puzzles

GAME = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nd4", "Nxe5", "Qg5", "Nxf7", "Qxg2"]


def fen_after(sans):
    board = chess.Board()
    for s in sans:
        board.push_san(s)
    return board.fen()


def row(pid="p1", rating=1500, rd=80, fen=None, moves="g5g2 h1f1", game="abcd1234", ply=9):
    return {"PuzzleId": pid, "FEN": fen or fen_after(GAME[:ply]), "Moves": moves, "Rating": str(rating),
            "RatingDeviation": str(rd), "GameUrl": f"https://lichess.org/game/{game}/black#{ply}"}


# --- parse_game_url --------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://lichess.org/game/787zsVup/black#48",  # the format in the 2026-11 puzzle CSV
    "https://lichess.org/game/787zsVup#48",
    "https://lichess.org/787zsVup/black#48",
    "https://lichess.org/787zsVup#48",
    "https://lichess.org/787zsVup",
])
def test_parse_game_url_extracts_the_game_id(url):
    assert parse_game_url(url) == "787zsVup"


# --- history_to_position ---------------------------------------------------------------------
def test_history_is_the_game_prefix_that_reaches_the_puzzle_fen():
    assert history_to_position(GAME, fen_after(GAME[:5])) == GAME[:5]


def test_history_can_be_empty_for_the_start_position():
    assert history_to_position(GAME, chess.STARTING_FEN) == []


def test_history_is_none_when_the_game_never_reaches_the_fen():
    other = fen_after(["d4", "d5"])
    assert history_to_position(GAME, other) is None


def test_history_ignores_move_clocks_and_en_passant_notation():
    # Lichess FENs may differ in clocks or in writing an en passant square nobody can use.
    fen = fen_after(GAME[:1])  # after 1.e4: python-chess omits "e3" (no legal ep capture)
    parts = fen.split()
    parts[3], parts[4], parts[5] = "e3", "7", "9"
    assert history_to_position(GAME, " ".join(parts)) == GAME[:1]


def test_history_respects_side_to_move():
    # Same piece placement, wrong side to move: not the puzzle position.
    parts = fen_after(GAME[:2]).split()
    parts[1] = "b"
    assert history_to_position(GAME, " ".join(parts)) is None


# --- select_puzzles --------------------------------------------------------------------------
def many_rows():
    return [row(pid=f"p{i}", rating=r, rd=rd) for i, (r, rd) in
            enumerate((r, rd) for r in range(600, 3000, 3) for rd in (70, 120))]


def test_select_takes_per_bucket_puzzles_from_each_rating_bucket():
    picked = select_puzzles(many_rows(), per_bucket=7)
    edges = (1000, 1500, 2000, 2500)
    counts = Counter(sum(int(p["Rating"]) >= e for e in edges) for p in picked)
    assert counts == {0: 7, 1: 7, 2: 7, 3: 7, 4: 7}


def test_select_drops_unsettled_ratings():
    assert all(int(p["RatingDeviation"]) <= 90 for p in select_puzzles(many_rows(), per_bucket=7))


def test_select_is_deterministic_for_a_seed_and_varies_across_seeds():
    ids = lambda seed: [p["PuzzleId"] for p in select_puzzles(many_rows(), per_bucket=7, seed=seed)]
    assert ids(0) == ids(0)
    assert ids(0) != ids(1)


def test_select_is_not_just_the_first_rows():
    picked = {p["PuzzleId"] for p in select_puzzles(many_rows(), per_bucket=7)}
    first_per_bucket = {"p0", "p2", "p4", "p6", "p8", "p10", "p12"}  # first 7 settled <1000 rows
    assert picked & first_per_bucket != first_per_bucket


# --- fetch_games -----------------------------------------------------------------------------
def ndjson(ids):
    return "\n".join(json.dumps({"id": i, "variant": "standard", "moves": "e4 e5"}) for i in ids) + "\n"


def test_fetch_batches_ids_and_parses_moves():
    calls = []

    def post(url, data, headers):
        calls.append(data.split(","))
        return 200, ndjson(data.split(","))
    games = fetch_games([f"g{i}" for i in range(7)], post, batch_size=3, sleep=lambda s: None)
    assert [len(c) for c in calls] == [3, 3, 1]
    assert games == {f"g{i}": ["e4", "e5"] for i in range(7)}


def test_fetch_waits_and_retries_on_rate_limit():
    replies = iter([(429, ""), (200, ndjson(["g0"]))])
    slept = []
    games = fetch_games(["g0"], lambda u, d, h: next(replies), sleep=slept.append)
    assert games == {"g0": ["e4", "e5"]}
    assert slept and slept[0] >= 60


def test_fetch_skips_games_that_are_not_standard_chess():
    body = (json.dumps({"id": "g0", "variant": "standard", "moves": "e4"}) + "\n"
            + json.dumps({"id": "g1", "variant": "fromPosition", "moves": "Nf3"}) + "\n")
    assert fetch_games(["g0", "g1"], lambda u, d, h: (200, body), sleep=lambda s: None) == {"g0": ["e4"]}


def test_fetch_fails_loudly_on_other_http_errors():
    with pytest.raises(RuntimeError, match="HTTP 500"):
        fetch_games(["g0"], lambda u, d, h: (500, "oops"), sleep=lambda s: None)


# --- build_puzzles ---------------------------------------------------------------------------
def test_build_puzzles_attaches_history_and_moves():
    puzzles, dropped = build_puzzles([row()], {"abcd1234": GAME})
    assert len(puzzles) == 1 and not dropped
    p = puzzles[0]
    assert (p.id, p.rating, p.history, p.moves) == ("p1", 1500, GAME[:9], ["g5g2", "h1f1"])


def test_build_puzzles_counts_why_puzzles_are_dropped():
    rows = [row(pid="a", game="missing"), row(pid="b", fen=fen_after(["d4"]))]
    puzzles, dropped = build_puzzles(rows, {"abcd1234": GAME})
    assert puzzles == []
    assert dropped == Counter({"game_not_fetched": 1, "fen_not_reached": 1})
