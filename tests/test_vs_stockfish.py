"""Tests for eval/vs_stockfish.py.

The game loop is tested with scripted players (fixed SAN lists), so every result is known
exactly. The model wrapper uses UniformModel, whose greedy output is the delimiter (an empty,
illegal move). The Stockfish wrapper runs only if a stockfish binary is found.
"""

import shutil
from pathlib import Path

import chess
import chess.engine
import pytest
import torch

from chessgpt.player import UniformModel
from chessgpt.tokenizer import CharTokenizer
from eval.vs_stockfish import model_player, play_game, stockfish_player

TOK = CharTokenizer.from_text(" #+-./0123456789=BKNOQRabcdefghx", extra="\n")


def scripted(sans, calls=None):
    """Player that plays the given SAN moves in order and records the history it was shown."""
    it = iter(sans)

    def play(board, history):
        if calls is not None:
            calls.append(list(history))
        return board.parse_san(next(it)), 0
    return play


def forfeiting(illegal=3):
    return lambda board, history: (None, illegal)


KNIGHTS_W = ["Nf3", "Ng1"] * 10
KNIGHTS_B = ["Nf6", "Ng8"] * 10


def test_checkmate_ends_game_with_winner():
    rec = play_game(scripted(["f3", "g4"]), scripted(["e5", "Qh4#"]))
    assert rec.result == "0-1"
    assert rec.termination == "checkmate"
    assert rec.sans == ["f3", "e5", "g4", "Qh4#"]
    assert rec.forfeit_by is None


def test_players_are_called_on_their_own_turns_with_the_san_history():
    w_calls, b_calls = [], []
    play_game(scripted(["f3", "g4"], w_calls), scripted(["e5", "Qh4#"], b_calls))
    assert w_calls == [[], ["f3", "e5"]]
    assert b_calls == [["f3"], ["f3", "e5", "g4"]]


@pytest.mark.parametrize("side,expected", [("white", "0-1"), ("black", "1-0")])
def test_forfeit_is_a_loss_for_the_side_that_could_not_move(side, expected):
    other = scripted(["e4"] if side == "black" else ["e5"])
    white, black = (forfeiting(), other) if side == "white" else (other, forfeiting())
    rec = play_game(white, black)
    assert rec.result == expected
    assert rec.termination == "forfeit"
    assert rec.forfeit_by == side
    assert rec.illegal_tries[side] == 3


def test_illegal_tries_are_summed_per_side():
    def white(board, history):
        return board.parse_san(["f3", "g4"][len(history) // 2]), 2
    rec = play_game(white, scripted(["e5", "Qh4#"]))
    assert rec.illegal_tries == {"white": 4, "black": 0}


def test_threefold_repetition_is_claimed_automatically():
    rec = play_game(scripted(KNIGHTS_W), scripted(KNIGHTS_B))
    assert rec.result == "1/2-1/2"
    assert rec.termination == "threefold_repetition"
    assert len(rec.sans) == 8  # start position seen for the 3rd time after 8 plies


def test_max_plies_adjudicates_a_draw():
    rec = play_game(scripted(KNIGHTS_W), scripted(KNIGHTS_B), max_plies=4)
    assert rec.result == "1/2-1/2"
    assert rec.termination == "max_plies"
    assert len(rec.sans) == 4


class FixedFirstChar(UniformModel):
    """Writes nothing but `first_logits` at a move's first character, then "4", then a space:
    so it writes "<c>4 " or the delimiter alone (an empty, illegal move)."""

    def __init__(self, first_logits):
        super().__init__(TOK.vocab_size)
        self.first_logits = first_logits

    def forward(self, idx, targets=None):
        logits, _ = super().forward(idx, targets)
        logits[:] = -30.0
        last = TOK.itos[idx[0, -1].item()]
        if last == ".":
            for c, v in self.first_logits.items():
                logits[:, -1, TOK.stoi[c]] = v
        elif last == "e":
            logits[:, -1, TOK.stoi["4"]] = 30.0
        else:
            logits[:, -1, TOK.stoi[" "]] = 30.0
        return logits, None


def test_model_player_forfeits_after_all_retries_when_output_is_never_legal():
    player = model_player(FixedFirstChar({"\n": 30.0}), TOK, "white", max_retries=2)
    move, illegal = player(chess.Board(), [])
    assert move is None
    assert illegal == 3  # 1 try + 2 retries


def test_model_player_plays_the_greedy_move_first():
    player = model_player(FixedFirstChar({"e": 1.0, "\n": 0.9}), TOK, "white")
    move, illegal = player(chess.Board(), [])
    assert move == chess.Move.from_uci("e2e4")
    assert illegal == 0


def test_model_player_samples_its_retries_after_an_illegal_greedy_move():
    # Greedy writes the delimiter (illegal); retries at temperature 1 pick "e" about half the
    # time. Retrying greedily would repeat the same illegal output forever.
    gen = torch.Generator().manual_seed(0)
    player = model_player(FixedFirstChar({"\n": 1.0, "e": 0.9}), TOK, "white",
                          max_retries=20, generator=gen)
    move, illegal = player(chess.Board(), [])
    assert move == chess.Move.from_uci("e2e4")
    assert 1 <= illegal <= 20


def test_model_player_conditions_on_its_own_side_winning():
    seen = []

    class Spy(FixedFirstChar):
        def forward(self, idx, targets=None):
            seen.append(TOK.decode(idx[0].tolist()))
            return super().forward(idx, targets)
    board = chess.Board()
    board.push_san("e4")
    model_player(Spy({"e": 30.0}), TOK, "black")(board, ["e4"])
    model_player(Spy({"e": 30.0}), TOK, "white")(chess.Board(), [])
    assert seen[0].startswith("\n0-1 1.e4 ")
    assert seen[-1].startswith("\n1-0 1.")


STOCKFISH = shutil.which("stockfish") or next(
    (p for p in ("/run/host/usr/bin/stockfish", "/usr/games/stockfish") if Path(p).exists()), None)


@pytest.mark.skipif(STOCKFISH is None, reason="no stockfish binary")
def test_stockfish_player_plays_legal_moves_and_mates_in_one():
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH) as engine:
        player = stockfish_player(engine, skill=20, nodes=20_000)
        board = chess.Board()
        for san in ["f3", "e5", "g4"]:
            board.push_san(san)
        move, illegal = player(board, ["f3", "e5", "g4"])
        assert board.san(move) == "Qh4#"
        assert illegal == 0


@pytest.mark.skipif(STOCKFISH is None, reason="no stockfish binary")
def test_stockfish_player_sets_its_skill_level_before_each_move():
    # Several players share one engine process (one per level), so each must set its own level.
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH) as engine:
        weak, strong = stockfish_player(engine, skill=0, nodes=1_000), stockfish_player(engine, skill=2, nodes=1_000)
        strong(chess.Board(), [])
        weak(chess.Board(), [])
        assert engine.protocol.config["Skill Level"] == 0
