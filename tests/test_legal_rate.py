"""Tests for eval/legal_rate.py: loading val games and classifying illegal outputs.

Each position is built with python-chess, and each output was checked to be NOT the exact SAN
of a legal move, so classify_illegal is only ever asked about outputs the metric rejects.
"""

import chess
import numpy as np
import pytest

from chessgpt.tokenizer import CharTokenizer
from eval.legal_rate import classify_illegal, load_games

TOK = CharTokenizer.from_text(" #+-./0123456789=BKNOQRabcdefghx", extra="\n")


def board_after(*sans):
    b = chess.Board()
    for s in sans:
        b.push_san(s)
    return b


def test_unreachable_move_is_illegal_move():
    assert classify_illegal(chess.Board(), "Nf4") == "illegal_move"


def test_moving_a_pinned_piece_is_illegal_move():
    b = board_after("e4", "d6", "d4", "Nd7", "Bb5")  # Nd7 is pinned to the king by Bb5
    assert classify_illegal(b, "Ne5") == "illegal_move"


@pytest.mark.parametrize("text", ["Nxf3", "Nf3+", "g1f3"])  # phantom capture, phantom check, long algebraic
def test_a_real_move_written_wrong_is_notation(text):
    assert classify_illegal(chess.Board(), text) == "notation"


def test_missing_disambiguation_is_ambiguous():
    b = board_after("e4", "a6", "Nc3", "a5")  # Nc3 and Ng1 can both go to e2
    assert classify_illegal(b, "Ne2") == "ambiguous"


def test_unrelated_move_while_in_check_is_ignores_check():
    b = board_after("e4", "f5", "Qh5+")
    assert classify_illegal(b, "Nf6") == "ignores_check"


@pytest.mark.parametrize("text", ["", "e9", "--"])  # "--" is a null move to python-chess, not a move
def test_garbage_and_null_moves_are_not_a_move(text):
    assert classify_illegal(chess.Board(), text) == "not_a_move"


def test_load_games_splits_on_the_delimiter_and_drops_empties(tmp_path):
    text = "\n1-0 1.e4 e5\n0-1 1.d4 d5\n1/2-1/2 1.c4"
    path = tmp_path / "val.bin"
    np.array(TOK.encode(text), dtype=np.uint8).tofile(path)
    assert load_games(path, TOK) == ["1-0 1.e4 e5", "0-1 1.d4 d5", "1/2-1/2 1.c4"]
    assert load_games(path, TOK, n=2) == ["1-0 1.e4 e5", "0-1 1.d4 d5"]
