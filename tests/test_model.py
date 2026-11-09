"""Tests for chessgpt/model.py: the GPT interface contract and its key properties.

Expected values are derived by hand or with plain torch ops, never with model code.
"""

import math
import random

import chess
import pytest
import torch
import torch.nn.functional as F

from chessgpt.model import GPT, GPTConfig
from chessgpt.player import choose_move
from chessgpt.tokenizer import CharTokenizer

V, BLOCK, LAYERS, HEADS, EMBD = 11, 8, 2, 4, 16


def tiny(**overrides) -> GPT:
    torch.manual_seed(0)
    cfg = dict(vocab_size=V, block_size=BLOCK, n_layer=LAYERS, n_head=HEADS, n_embd=EMBD, dropout=0.0)
    cfg.update(overrides)
    return GPT(GPTConfig(**cfg))


def rand_idx(B=2, T=BLOCK, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, T), generator=g)


# --- interface ----------------------------------------------------------------
def test_forward_shapes_without_targets():
    logits, loss = tiny()(rand_idx(B=3, T=5))
    assert logits.shape == (3, 5, V)
    assert loss is None


def test_loss_is_mean_cross_entropy_of_returned_logits():
    model = tiny()
    idx, targets = rand_idx(seed=1), rand_idx(seed=2)
    logits, loss = model(idx, targets)
    expected = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
    assert loss.dim() == 0
    assert torch.allclose(loss, expected, atol=1e-6)


def test_initial_loss_is_close_to_uniform_guess():
    # Small init -> near-uniform predictions -> loss ~ ln(V) = 2.398 for V=11.
    _, loss = tiny()(rand_idx(seed=1), rand_idx(seed=2))
    assert abs(loss.item() - math.log(V)) < 0.1


def test_sequence_longer_than_block_size_is_rejected():
    with pytest.raises(ValueError):
        tiny()(torch.zeros((1, BLOCK + 1), dtype=torch.long))


def test_n_embd_not_divisible_by_n_head_is_rejected():
    with pytest.raises(ValueError):
        tiny(n_embd=18, n_head=4)


# --- causality and context ------------------------------------------------------
def test_future_tokens_do_not_affect_past_logits():
    model = tiny().eval()
    idx = rand_idx()
    changed = idx.clone()
    changed[:, 5] = (changed[:, 5] + 1) % V
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6), "positions < 5 saw token 5: causal mask broken"


def test_past_tokens_do_affect_later_logits():
    model = tiny().eval()
    idx = rand_idx()
    changed = idx.clone()
    changed[:, 5] = (changed[:, 5] + 1) % V
    a, _ = model(idx)
    b, _ = model(changed)
    assert not torch.allclose(a[:, 6:], b[:, 6:], atol=1e-4), "later positions ignore earlier tokens"


def test_prefix_logits_match_full_sequence_logits():
    model = tiny().eval()
    idx = rand_idx()
    full, _ = model(idx)
    prefix, _ = model(idx[:, :3])
    assert torch.allclose(full[:, :3], prefix, atol=1e-6)


def test_manual_and_fused_attention_agree():
    fused = tiny(attn_impl="sdpa").eval()
    manual = tiny(attn_impl="manual").eval()
    manual.load_state_dict(fused.state_dict())
    idx = rand_idx()
    assert torch.allclose(fused(idx)[0], manual(idx)[0], atol=1e-5)


def test_eval_mode_is_deterministic_with_dropout():
    model = tiny(dropout=0.5)
    idx = rand_idx()
    model.train()
    t1, _ = model(idx)
    t2, _ = model(idx)
    assert not torch.allclose(t1, t2), "dropout should be active in train mode"
    model.eval()
    e1, _ = model(idx)
    e2, _ = model(idx)
    assert torch.equal(e1, e2), "dropout still active in eval mode"


# --- parameters and learning -------------------------------------------------------
def test_parameter_count_matches_hand_derivation():
    # Per block: 2 LayerNorms (2*2d) + qkv (3d^2+3d) + attn proj (d^2+d)
    #            + MLP fc (4d^2+4d) + MLP proj (4d^2+d) = 12d^2 + 13d = 3280 for d=16.
    # Total: 2 blocks 6560 + token emb 11*16=176 + pos emb 8*16=128 + final LN 32 = 6896.
    # lm_head is tied to the token embedding, so it adds nothing.
    model = tiny()
    assert model.num_params(non_embedding=False) == 6896
    assert model.num_params() == 6896 - 128
    assert sum(p.numel() for p in model.parameters()) == 6896


def test_overfits_a_single_sequence():
    torch.manual_seed(0)
    model = tiny()
    seq = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5]])
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(200):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 < first


# --- integration with the player ------------------------------------------------------
@pytest.fixture
def tok():
    return CharTokenizer.from_text(" #+-./0123456789=BKNOQRabcdefghx", extra="\n")


def test_choose_move_runs_with_real_gpt(tok):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=16))
    choice = choose_move(model, tok, ["e4", "e5"], max_retries=2, result_prefix="1-0",
                         generator=torch.Generator().manual_seed(0))
    probs = [c.prob for c in choice.candidates]
    assert len(choice.candidates) == 5
    assert probs == sorted(probs, reverse=True)
    assert 0 < choice.legal_mass <= 1


def test_choose_move_handles_game_longer_than_block_size(tok):
    rng = random.Random(0)
    board, moves = chess.Board(), []
    for _ in range(40):  # 40 plies of movetext is > 64 characters
        move = rng.choice(list(board.legal_moves))
        moves.append(board.san(move))
        board.push(move)
    assert not board.is_game_over()
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=16))
    choice = choose_move(model, tok, moves, max_retries=0, result_prefix="1-0")
    assert len(choice.candidates) == 5
