"""Tests for chessgpt/train.py: LR schedule, optimizer groups, batching, the
training step, and the full loop incl. checkpoint/resume.

Expected values are hand-derived; the data fixture is a tiny repetitive "chess" text.
"""

import copy
import csv

import pytest
import torch

from chessgpt.model import GPT, GPTConfig
from chessgpt.tokenizer import CharTokenizer
from chessgpt.train import (
    configure_optimizer,
    estimate_loss,
    get_batch,
    get_lr,
    train,
    train_step,
)

# --- fixtures -------------------------------------------------------------------------
GAME = "\n1-0 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7"


@pytest.fixture
def data_dir(tmp_path):
    """train.bin / val.bin / meta.json built from a repeated game."""
    tok = CharTokenizer.from_text(GAME)
    d = tmp_path / "data"
    d.mkdir()
    tok.encode_array(GAME * 200).tofile(d / "train.bin")
    tok.encode_array(GAME * 20).tofile(d / "val.bin")
    tok.save(d / "meta.json", delimiter="\n")
    return d


def tiny_cfg(**train_overrides):
    train_cfg = dict(
        batch_size=4, grad_accum_steps=1, max_iters=40, lr=1e-2, min_lr=1e-3,
        warmup_iters=5, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
        eval_interval=10, eval_iters=2, ckpt_interval=10, fp16=False, seed=1337,
    )
    train_cfg.update(train_overrides)
    return {
        "model": dict(vocab_size=None, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.1),
        "train": train_cfg,
    }


def read_log(out_dir):
    with open(out_dir / "log.csv") as f:
        return list(csv.DictReader(f))


# --- learning-rate schedule ------------------------------------------------------------
@pytest.mark.parametrize("it, expected", [
    (0, 0.1),     # warmup: lr * (it+1) / warmup_iters
    (4, 0.5),
    (9, 1.0),     # last warmup step reaches the peak
    (10, 1.0),    # cosine starts at the peak
    (60, 0.55),   # halfway through decay: min + 0.5 * (lr - min) = 0.1 + 0.45
    (110, 0.1),   # end of decay: min_lr
    (500, 0.1),   # after max_iters: stays at min_lr
])
def test_lr_schedule(it, expected):
    cfg = dict(lr=1.0, min_lr=0.1, warmup_iters=10, max_iters=110)
    assert get_lr(it, cfg) == pytest.approx(expected)


# --- optimizer ------------------------------------------------------------------------
def test_optimizer_decays_matrices_but_not_biases_or_layernorms():
    model = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=2, n_embd=8))
    opt = configure_optimizer(model, tiny_cfg()["train"])
    decay = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    no_decay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0 for p in g["params"]}

    assert decay.isdisjoint(no_decay)
    assert decay | no_decay == {id(p) for p in model.parameters()}
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))
    assert id(model.blocks[0].attn.c_attn.weight) in decay
    assert id(model.wte.weight) in decay
    assert id(model.blocks[0].attn.c_attn.bias) in no_decay
    assert id(model.blocks[0].ln_1.weight) in no_decay
    assert {g["weight_decay"] for g in opt.param_groups} == {0.1, 0.0}


# --- batching -----------------------------------------------------------------------------
def test_get_batch_with_generator_is_reproducible_and_leaves_global_rng_alone(data_dir):
    state = torch.get_rng_state()
    x1, y1 = get_batch(data_dir, "val", 4, 8, generator=torch.Generator().manual_seed(0))
    x2, y2 = get_batch(data_dir, "val", 4, 8, generator=torch.Generator().manual_seed(0))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(x1[:, 1:], y1[:, :-1])


def test_estimate_loss_restores_train_mode(data_dir):
    model = GPT(GPTConfig(vocab_size=CharTokenizer.load(data_dir / "meta.json").vocab_size,
                          block_size=16, n_layer=1, n_head=2, n_embd=16))
    model.train()
    losses = estimate_loss(model, data_dir, block_size=16, batch_size=4, eval_iters=2, device="cpu")
    assert model.training
    assert set(losses) == {"train", "val"}
    assert all(isinstance(v, float) and v > 0 for v in losses.values())


# --- training step --------------------------------------------------------------------
def test_grad_accumulation_matches_one_big_batch(data_dir):
    torch.manual_seed(0)
    a = GPT(GPTConfig(vocab_size=CharTokenizer.load(data_dir / "meta.json").vocab_size,
                      block_size=16, n_layer=1, n_head=2, n_embd=16))
    b = copy.deepcopy(a)
    g = torch.Generator().manual_seed(0)
    x1, y1 = get_batch(data_dir, "train", 4, 16, generator=g)
    x2, y2 = get_batch(data_dir, "train", 4, 16, generator=g)

    opt_a = torch.optim.SGD(a.parameters(), lr=1.0)
    opt_b = torch.optim.SGD(b.parameters(), lr=1.0)
    train_step(a, opt_a, None, [(x1, y1), (x2, y2)], grad_clip=0.0)
    train_step(b, opt_b, None, [(torch.cat([x1, x2]), torch.cat([y1, y2]))], grad_clip=0.0)

    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), name


# --- full loop ------------------------------------------------------------------------------
def test_training_run_reduces_val_loss_and_writes_outputs(data_dir, tmp_path):
    # 120 iterations: with 40 the cosine schedule reaches min_lr while val loss is
    # still ~1.75 (measured), so "halve the loss" needs a longer run, not a looser bar.
    out = tmp_path / "run"
    summary = train(tiny_cfg(max_iters=120, eval_interval=30, ckpt_interval=30), data_dir, out)

    rows = read_log(out)
    assert [int(r["iter"]) for r in rows] == [0, 30, 60, 90, 120]
    assert float(rows[-1]["val_loss"]) < 0.5 * float(rows[0]["val_loss"])
    assert int(rows[-1]["tokens_seen"]) == 120 * 4 * 16   # iters * batch * block
    assert (out / "ckpt.pt").exists() and (out / "best.pt").exists()
    assert summary["iter"] == 120


def test_resume_is_identical_to_uninterrupted_run(data_dir, tmp_path):
    full, split = tmp_path / "full", tmp_path / "split"
    train(tiny_cfg(), data_dir, full)
    train(tiny_cfg(), data_dir, split, iters_this_run=20)
    assert torch.load(split / "ckpt.pt")["iter"] == 20
    train(tiny_cfg(), data_dir, split, resume=True)

    sd_full = torch.load(full / "ckpt.pt")["model"]
    sd_split = torch.load(split / "ckpt.pt")["model"]
    for k in sd_full:
        assert torch.equal(sd_full[k], sd_split[k]), k

    def strip(rows):
        return [{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows]
    assert strip(read_log(full)) == strip(read_log(split))


def test_time_limit_stops_early_and_saves_a_checkpoint(data_dir, tmp_path):
    out = tmp_path / "run"
    summary = train(tiny_cfg(), data_dir, out, max_seconds=0)
    assert summary["iter"] == 1
    assert torch.load(out / "ckpt.pt")["iter"] == 1


def test_resume_without_checkpoint_fails(data_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        train(tiny_cfg(), data_dir, tmp_path / "empty", resume=True)


# --- fine-tuning: init_from (decision 012) ----------------------------------------------------
def test_init_from_starts_from_the_source_weights(data_dir, tmp_path):
    train(tiny_cfg(), data_dir, tmp_path / "src")
    # lr 0: AdamW changes nothing (decoupled weight decay is scaled by lr too), so the result
    # equals whatever the run started from: the source weights, not a fresh random init.
    train(tiny_cfg(lr=0.0, min_lr=0.0, max_iters=5), data_dir, tmp_path / "ft",
          init_from=tmp_path / "src" / "best.pt")
    src = torch.load(tmp_path / "src" / "best.pt")["model"]
    ft = torch.load(tmp_path / "ft" / "ckpt.pt")["model"]
    for k in src:
        assert torch.equal(src[k], ft[k]), k


def test_init_from_takes_the_architecture_from_the_source(data_dir, tmp_path):
    train(tiny_cfg(max_iters=10), data_dir, tmp_path / "src")
    cfg = tiny_cfg(max_iters=5)
    cfg["model"]["n_layer"] = 3  # ignored: the weights decide the architecture
    train(cfg, data_dir, tmp_path / "ft", init_from=tmp_path / "src" / "best.pt")
    assert torch.load(tmp_path / "ft" / "best.pt")["model_args"]["n_layer"] == 1


def test_init_from_is_a_fresh_run_with_its_own_schedule(data_dir, tmp_path):
    train(tiny_cfg(), data_dir, tmp_path / "src")          # source ran 40 iterations
    summary = train(tiny_cfg(max_iters=20), data_dir, tmp_path / "ft", init_from=tmp_path / "src" / "best.pt")
    assert [int(r["iter"]) for r in read_log(tmp_path / "ft")] == [0, 10, 20]
    assert summary["iter"] == 20


def test_init_from_rejects_a_checkpoint_with_another_vocabulary(data_dir, tmp_path):
    train(tiny_cfg(max_iters=10), data_dir, tmp_path / "src")
    other = tmp_path / "other"
    other.mkdir()
    tok = CharTokenizer.from_text(GAME + "xyz")
    for f in ("train.bin", "val.bin"):
        tok.encode_array(GAME * 50).tofile(other / f)
    tok.save(other / "meta.json", delimiter="\n")
    with pytest.raises(ValueError, match="vocab"):
        train(tiny_cfg(max_iters=5), other, tmp_path / "ft", init_from=tmp_path / "src" / "best.pt")


def test_init_from_and_resume_cannot_be_combined(data_dir, tmp_path):
    train(tiny_cfg(max_iters=10), data_dir, tmp_path / "src")
    with pytest.raises(ValueError, match="init_from"):
        train(tiny_cfg(), data_dir, tmp_path / "src", resume=True, init_from=tmp_path / "src" / "best.pt")
