"""Tests for chessgpt/player.py: what counts as a legal move.

The model's text is legal only if it is EXACTLY the SAN of a legal move. python-chess's
parse_san is lenient (it accepts phantom captures/checks, long algebraic, and "--" as a
null move), which would make Python and the JS port (chess.js) disagree.
"""

import chess
import pytest
import torch

from chessgpt.model import GPTConfig
from chessgpt.player import choose_move, legal_move_from_text
from chessgpt.tokenizer import CharTokenizer

TOK = CharTokenizer.from_text(" #+-./0123456789=BKNOQRabcdefghx", extra="\n")


class OracleModel(torch.nn.Module):
    """Puts all probability on the character following the input prefix in `text`."""

    def __init__(self, text: str):
        super().__init__()
        self.config = GPTConfig(vocab_size=TOK.vocab_size, block_size=512)
        self.text = text

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.full((B, T, TOK.vocab_size), -20.0)
        for b in range(B):
            seen = TOK.decode(idx[b].tolist())
            for t in range(T):
                prefix = seen[: t + 1]
                ok = self.text.startswith(prefix) and len(prefix) < len(self.text)
                logits[b, t, TOK.stoi[self.text[len(prefix)] if ok else "\n"]] = 20.0
        return logits, None


@pytest.fixture
def board():
    b = chess.Board()
    b.push_san("e4")
    b.push_san("e5")
    return b


@pytest.mark.parametrize("text", ["Nf3", "Nc3", "Qh5", "Bc4"])
def test_exact_san_of_a_legal_move_is_accepted(board, text):
    move = legal_move_from_text(board, text)
    assert move is not None and board.san(move) == text


@pytest.mark.parametrize("text, why", [
    ("Nxf3", "phantom capture"),
    ("Nf3+", "phantom check"),
    ("Ng1f3", "long algebraic, not SAN"),
    ("--", "null move"),
    ("", "empty output"),
    ("Nf6", "illegal for White"),
    ("e4e5", "garbage"),
])
def test_anything_but_exact_san_is_rejected(board, text, why):
    assert legal_move_from_text(board, text) is None, why


def test_missing_check_symbol_is_rejected():
    b = chess.Board()
    for san in ["e4", "f5", "d3"]:
        b.push_san(san)
    b.push_san("g5")
    assert b.san(b.parse_san("Qh5")) == "Qh5#"
    assert legal_move_from_text(b, "Qh5") is None
    assert legal_move_from_text(b, "Qh5#") is not None


def test_choose_move_counts_a_phantom_capture_as_an_illegal_try():
    model = OracleModel("\n1-0 1.e4 e5 2.Nxf3 ")
    c = choose_move(model, TOK, ["e4", "e5"], temperature=0.0, max_retries=1, result_prefix="1-0")
    assert c.move is None
    assert c.illegal_tries == ["Nxf3", "Nxf3"]


def test_choose_move_never_plays_a_null_move():
    model = OracleModel("\n1-0 1.e4 e5 2.-- ")
    c = choose_move(model, TOK, ["e4", "e5"], temperature=0.0, max_retries=0, result_prefix="1-0")
    assert c.move is None and c.illegal_tries == ["--"]
