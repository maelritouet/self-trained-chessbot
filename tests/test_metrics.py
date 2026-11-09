"""Tests for eval/metrics.py.

Generation is tested with an *oracle* model that knows one game text and always predicts
its next character, so the expected outputs are exact. The metric functions themselves
are pure and tested on hand-written records.
"""

import pytest
import torch

from chessgpt.model import GPTConfig
from chessgpt.player import UniformModel
from chessgpt.tokenizer import CharTokenizer
from eval.metrics import (
    PositionRecord,
    Puzzle,
    evaluate_positions,
    legal_move_rate,
    legal_rate_by_move_number,
    move_match_rate,
    puzzle_accuracy,
    score_vs_engine,
    solve_puzzle,
)

TOK = CharTokenizer.from_text(" #+-./0123456789=BKNOQRabcdefghx", extra="\n")


class OracleModel(torch.nn.Module):
    """GPT-interface stand-in: at each position, puts all probability on the character that
    follows the input prefix in `text` (or on the delimiter once the text is exhausted)."""

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
                if self.text.startswith(prefix) and len(prefix) < len(self.text):
                    nxt = self.text[len(prefix)]
                else:
                    nxt = "\n"
                logits[b, t, TOK.stoi[nxt]] = 20.0
        return logits, None


GAME = "1-0 1.e4 e5 2.Nf3 Nc6 3.Bb5"


def rec(move_number, legal, match=False, ply=0, game=0):
    return PositionRecord(game=game, ply=ply, move_number=move_number,
                          output="x", legal=legal, matches_game=match)


# --- generation over real games ----------------------------------------------------------
def test_oracle_of_the_same_game_is_legal_and_matches_everywhere():
    records = evaluate_positions(OracleModel("\n" + GAME + "\n"), TOK, [GAME])
    assert [r.ply for r in records] == [0, 1, 2, 3, 4]
    assert [r.move_number for r in records] == [1, 1, 2, 2, 3]
    assert [r.output for r in records] == ["e4", "e5", "Nf3", "Nc6", "Bb5"]
    assert all(r.legal and r.matches_game for r in records)


def test_legal_but_different_move_is_legal_without_matching():
    oracle = OracleModel("\n1-0 1.e4 e5 2.Nf3 Nc6 3.Bc4\n")   # plays 3.Bc4 where the game had 3.Bb5
    last = evaluate_positions(oracle, TOK, [GAME])[-1]
    assert last.output == "Bc4"
    assert last.legal and not last.matches_game


def test_illegal_output_is_recorded_as_illegal():
    oracle = OracleModel("\n1-0 1.e4 e5 2.Nf3 Nc6 3.Bb6\n")   # Bb6 is not a legal move there
    last = evaluate_positions(oracle, TOK, [GAME])[-1]
    assert last.output == "Bb6"
    assert not last.legal and not last.matches_game


def test_phantom_capture_is_not_a_legal_output():
    # 3.Bxb5 claims a capture on an empty square: python-chess's lenient parser would accept it.
    oracle = OracleModel("\n1-0 1.e4 e5 2.Nf3 Nc6 3.Bxb5\n")
    last = evaluate_positions(oracle, TOK, [GAME])[-1]
    assert last.output == "Bxb5"
    assert not last.legal and not last.matches_game


def test_uniform_model_produces_no_legal_moves():
    records = evaluate_positions(UniformModel(TOK.vocab_size), TOK, [GAME])
    assert len(records) == 5
    assert not any(r.legal for r in records)


def test_max_plies_limits_positions_per_game():
    records = evaluate_positions(OracleModel("\n" + GAME + "\n"), TOK, [GAME, GAME], max_plies=2)
    assert [(r.game, r.ply) for r in records] == [(0, 0), (0, 1), (1, 0), (1, 1)]


# --- metrics over records -------------------------------------------------------------------
def test_legal_move_rate():
    assert legal_move_rate([rec(1, True), rec(1, True), rec(2, True), rec(2, False)]) == 0.75


def test_rates_on_no_records_are_rejected():
    with pytest.raises(ValueError):
        legal_move_rate([])
    with pytest.raises(ValueError):
        move_match_rate([])


def test_move_match_rate():
    assert move_match_rate([rec(1, True, True), rec(1, True, False), rec(2, False)]) == pytest.approx(1 / 3)


def test_legal_rate_by_move_number_buckets_by_first_move_of_bucket():
    records = [rec(1, True), rec(10, False), rec(11, True), rec(20, True), rec(25, False)]
    assert legal_rate_by_move_number(records, bucket_size=10) == {1: 0.5, 11: 1.0, 21: 0.0}


def test_legal_rate_by_move_number_drops_sparse_buckets():
    records = [rec(1, True), rec(2, True), rec(11, False)]
    assert legal_rate_by_move_number(records, bucket_size=10, min_count=2) == {1: 1.0}


# --- puzzles ----------------------------------------------------------------------------------
# After 1.c3 e6 2.a4 Qf6 3.g3 Bc5 4.Ra3, Black has two mates in one: Qxf2# and Bxf2#.
TWO_MATES = Puzzle(id="t1", rating=1200, history=["c3", "e6", "a4", "Qf6", "g3", "Bc5"],
                   moves=["a1a3", "f6f2"])
TWO_MOVE = Puzzle(id="t2", rating=900, history=[], moves=["e2e4", "e7e5", "g1f3", "b8c6"])


def test_puzzle_solved_with_the_expected_move():
    oracle = OracleModel("\n0-1 1.c3 e6 2.a4 Qf6 3.g3 Bc5 4.Ra3 Qxf2#\n")
    assert solve_puzzle(oracle, TOK, TWO_MATES) is True


def test_puzzle_solved_with_an_alternative_checkmate():
    oracle = OracleModel("\n0-1 1.c3 e6 2.a4 Qf6 3.g3 Bc5 4.Ra3 Bxf2#\n")
    assert solve_puzzle(oracle, TOK, TWO_MATES) is True


def test_puzzle_failed_with_a_legal_non_mating_move():
    oracle = OracleModel("\n0-1 1.c3 e6 2.a4 Qf6 3.g3 Bc5 4.Ra3 Qxb2\n")
    assert solve_puzzle(oracle, TOK, TWO_MATES) is False


def test_multi_move_puzzle_needs_every_model_move():
    # Model is Black: plays e5 after the setup 1.e4, then Nc6 after the forced reply 2.Nf3.
    assert solve_puzzle(OracleModel("\n0-1 1.e4 e5 2.Nf3 Nc6\n"), TOK, TWO_MOVE) is True
    assert solve_puzzle(OracleModel("\n0-1 1.e4 e5 2.Nf3 Nf6\n"), TOK, TWO_MOVE) is False


def test_puzzle_failed_on_illegal_output():
    assert solve_puzzle(OracleModel("\n0-1 1.e4 e4\n"), TOK, TWO_MOVE) is False


def test_puzzle_accuracy_by_rating_bucket():
    results = [(800, True), (950, False), (1200, True), (1499, True), (2600, False)]
    assert puzzle_accuracy(results) == {
        "<1000": 0.5, "1000-1499": 1.0, "2500+": 0.0, "all": pytest.approx(3 / 5),
    }


# --- games vs an engine -------------------------------------------------------------------------
def test_score_vs_engine_counts_from_the_models_side():
    s = score_vs_engine(["1-0", "1-0", "1/2-1/2", "0-1"], ["white", "black", "white", "white"])
    assert s == {"games": 4, "wins": 1, "draws": 1, "losses": 2, "forfeits": 0,
                 "score": pytest.approx(1.5 / 4)}


def test_forfeits_count_as_losses_and_are_reported():
    # The forfeited game's result string says the model (White) won: the forfeit must
    # override it. (A forfeit on a game already lost would not test the rule at all.)
    s = score_vs_engine(["1-0", "1-0"], ["white", "white"], forfeits=[False, True])
    assert (s["wins"], s["losses"], s["forfeits"]) == (1, 1, 1)
    assert s["score"] == 0.5


def test_score_vs_engine_rejects_bad_input():
    with pytest.raises(ValueError):
        score_vs_engine(["1-0"], ["white", "black"])
    with pytest.raises(ValueError):
        score_vs_engine(["*"], ["white"])
    with pytest.raises(ValueError):
        score_vs_engine(["1-0"], ["purple"])
