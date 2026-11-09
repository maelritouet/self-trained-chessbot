"""Training entry point:
    python -m chessgpt.train --config configs/base.yaml --data data/processed --out runs/x
    python -m chessgpt.train ... --resume              # continue from runs/x/ckpt.pt
    python -m chessgpt.train ... --max-hours 11.5      # stop + checkpoint before Kaggle's 12 h cap

Outputs in --out:
    log.csv     one row per evaluation: iter, lr, train_loss, val_loss, tokens_seen, elapsed_s
    ckpt.pt     latest full state, for resuming (model, optimizer, scaler, iter, RNG, config)
    best.pt     weights with the lowest val loss so far + model args + vocab (for evals / export)
    config.json the resolved config of the run

Map of one iteration:
    lr = warmup-then-cosine(it)
    every eval_interval: estimate train/val loss on fixed batches -> log.csv, best.pt
    grad_accum_steps micro-batches: forward under fp16 autocast, loss / steps, scaled backward
    unscale -> clip grad norm -> optimizer step (skipped by GradScaler on inf/NaN) -> scaler update
    every ckpt_interval (and when stopping): atomic save of ckpt.pt

Questions to be able to answer:
  - Why warmup? What happens to Adam's early updates without it?
  - Why does GradScaler exist, and why must gradients be unscaled before clipping?
  - What has to be saved so a resumed run is indistinguishable from an uninterrupted one?
  - Why divide the loss by grad_accum_steps?
  - Why no weight decay on biases and LayerNorm parameters?
  - Why do eval batches use their own generator instead of the global RNG?
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from chessgpt.model import GPT, GPTConfig

LOG_FIELDS = ["iter", "lr", "train_loss", "val_loss", "tokens_seen", "elapsed_s"]


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def get_batch(
    data_dir: str | Path, split: str, batch_size: int, block_size: int, device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows from {split}.bin.

    Returns:
        x: (batch_size, block_size) int64, tokens [i, i+block_size)
        y: (batch_size, block_size) int64, the same windows shifted by one: [i+1, i+block_size+1)

    generator: a CPU torch.Generator for the window offsets; None uses the global RNG.
    Windows may span several games; the delimiter token marks where a new one begins.
    The memmap is reopened on every call (as in nanoGPT): a long-lived np.memmap keeps
    growing in memory as pages are touched.
    """
    data = np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint8, mode="r")
    if len(data) <= block_size + 1:
        raise ValueError(f"{split}.bin has {len(data)} tokens, need > block_size + 1 = {block_size + 1}")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix.tolist()])
    if device.startswith("cuda"):
        # pinned memory + non_blocking lets the copy overlap with GPU compute
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_lr(it: int, train_cfg: dict) -> float:
    """Linear warmup to `lr` over warmup_iters, then cosine decay to `min_lr` at max_iters."""
    lr, min_lr = train_cfg["lr"], train_cfg["min_lr"]
    warmup, max_iters = train_cfg["warmup_iters"], train_cfg["max_iters"]
    if it < warmup:
        return lr * (it + 1) / warmup          # it=0 is already > 0, last warmup step hits lr
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)  # 0 -> 1 over the decay phase
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (lr - min_lr)


def configure_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices (Linear weights, embeddings) only.

    Biases and LayerNorm gains/biases are 1-D: decaying them toward 0 only fights
    the normalisation and adds no useful regularisation. model.parameters() yields
    the tied wte/lm_head matrix once, so it isn't double-counted.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": train_cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=train_cfg["lr"],
                             betas=(train_cfg["beta1"], train_cfg["beta2"]))


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, data_dir: str | Path, block_size: int, batch_size: int,
    eval_iters: int, device: str, seed: int = 0, use_fp16: bool = False,
) -> dict[str, float]:
    """Mean loss over eval_iters batches of each split, in eval mode (no dropout).

    Each split uses a fresh generator with the same seed, so every evaluation sees
    the same batches: val-loss points are comparable, and the global RNG that drives
    training (batch order, dropout) is untouched.
    """
    was_training = model.training
    model.eval()
    out = {}
    for split in ("train", "val"):
        g = torch.Generator().manual_seed(seed)
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data_dir, split, batch_size, block_size, device, generator=g)
            with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_fp16):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(sum(losses) / len(losses))
    model.train(was_training)
    return out


def train_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None, micro_batches: list[tuple[torch.Tensor, torch.Tensor]],
    grad_clip: float = 0.0, use_fp16: bool = False,
) -> float:
    """One optimizer step over several micro-batches (gradient accumulation).

    Each micro-batch loss is divided by the number of micro-batches, so the summed
    gradients equal the gradient of the mean loss over the whole (big) batch.
    Returns the mean loss.
    """
    optimizer.zero_grad(set_to_none=True)
    n = len(micro_batches)
    total = 0.0
    for x, y in micro_batches:
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_fp16):
            _, loss = model(x, y)
        loss = loss / n
        (scaler.scale(loss) if scaler is not None else loss).backward()
        total += loss.item()

    if scaler is not None:
        scaler.unscale_(optimizer)  # clip the true gradients, not the scaled ones
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if scaler is not None:
        scaler.step(optimizer)      # skips the step if any gradient is inf/NaN
        scaler.update()             # adapts the scale factor
    else:
        optimizer.step()
    return total


# --- checkpoints and logging ------------------------------------------------------------
def _atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename: a run killed mid-save never corrupts ckpt.pt."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _init_log(path: Path, keep_before_iter: int | None) -> None:
    """Fresh run: new log with header. Resume: drop rows logged after the checkpoint
    (a run killed between an eval and the next checkpoint would otherwise duplicate them)."""
    rows = []
    if keep_before_iter is not None and path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["iter"]) < keep_before_iter]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _append_log(path: Path, row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def train(
    cfg: dict, data_dir: str | Path, out_dir: str | Path, resume: bool = False,
    iters_this_run: int | None = None, max_seconds: float | None = None, device: str | None = None,
) -> dict:
    """Run (or resume) training. Returns {"iter", "best_val", "stopped_early"}.

    iters_this_run: stop after this many iterations in this session (checkpoint saved).
    max_seconds:    stop once this much wall time has passed in this session (checkpoint saved).
    """
    cfg = copy.deepcopy(cfg)
    mcfg, tcfg = cfg["model"], cfg["train"]
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(tcfg.get("fp16")) and device.startswith("cuda")

    meta = json.loads((data_dir / "meta.json").read_text())
    if mcfg.get("vocab_size") is None:
        mcfg["vocab_size"] = meta["vocab_size"]
    elif mcfg["vocab_size"] != meta["vocab_size"]:
        raise ValueError(f"config vocab_size {mcfg['vocab_size']} != meta.json {meta['vocab_size']}")

    ckpt_path, log_path = out_dir / "ckpt.pt", out_dir / "log.csv"
    ckpt = None
    if resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume given but no checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mcfg = ckpt["config"]["model"]  # architecture must match the saved weights
        if ckpt["config"]["train"] != tcfg:
            print("warning: train config differs from the checkpoint's; using the new one")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"model": mcfg, "train": tcfg}, indent=2) + "\n")

    torch.manual_seed(tcfg["seed"])
    model = GPT(GPTConfig(**mcfg)).to(device)
    optimizer = configure_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    start_iter, best_val, elapsed_before = 0, float("inf"), 0.0

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_iter, best_val, elapsed_before = ckpt["iter"], ckpt["best_val"], ckpt["elapsed_s"]
        # Restore RNG *after* building the model (its init consumed random numbers),
        # so batches and dropout masks continue exactly as in an uninterrupted run.
        torch.set_rng_state(ckpt["rng"]["torch"])
        if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    _init_log(log_path, start_iter if resume else None)

    B, T, accum = tcfg["batch_size"], mcfg["block_size"], tcfg["grad_accum_steps"]
    tokens_per_iter = B * accum * T
    print(f"device {device} | fp16 {use_fp16} | params {model.num_params():,} | "
          f"tokens/iter {tokens_per_iter:,} | start iter {start_iter}")

    t0 = time.time()

    def elapsed() -> float:
        return elapsed_before + time.time() - t0

    def evaluate(it: int) -> None:
        nonlocal best_val
        losses = estimate_loss(model, data_dir, T, B, tcfg["eval_iters"], device,
                               seed=tcfg["seed"], use_fp16=use_fp16)
        lr = get_lr(it, tcfg)
        _append_log(log_path, {
            "iter": it, "lr": f"{lr:.6g}", "train_loss": f"{losses['train']:.6f}",
            "val_loss": f"{losses['val']:.6f}", "tokens_seen": it * tokens_per_iter,
            "elapsed_s": f"{elapsed():.1f}",
        })
        print(f"iter {it:>7} | lr {lr:.2e} | train {losses['train']:.4f} | "
              f"val {losses['val']:.4f} | {elapsed() / 60:.1f} min")
        if losses["val"] < best_val:
            best_val = losses["val"]
            _atomic_save({
                "model": model.state_dict(), "model_args": mcfg, "iter": it, "val_loss": best_val,
                "itos": meta["itos"], "delimiter": meta.get("delimiter", "\n"),
            }, out_dir / "best.pt")

    def save_checkpoint(it: int) -> None:
        _atomic_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "iter": it,                     # the next iteration to run
            "best_val": best_val,
            "elapsed_s": elapsed(),
            "config": {"model": mcfg, "train": tcfg},
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }, ckpt_path)

    model.train()
    it, session_iters, stopped_early = start_iter, 0, False
    while it < tcfg["max_iters"]:
        lr = get_lr(it, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if it % tcfg["eval_interval"] == 0:
            evaluate(it)

        micro = [get_batch(data_dir, "train", B, T, device) for _ in range(accum)]
        train_step(model, optimizer, scaler, micro, tcfg["grad_clip"], use_fp16)
        it += 1
        session_iters += 1

        stopped_early = (
            (iters_this_run is not None and session_iters >= iters_this_run)
            or (max_seconds is not None and time.time() - t0 >= max_seconds)
        ) and it < tcfg["max_iters"]
        if it % tcfg["ckpt_interval"] == 0 or it == tcfg["max_iters"] or stopped_early:
            save_checkpoint(it)
        if stopped_early:
            print(f"stopping at iter {it} (session limit); resume with --resume")
            break

    if not stopped_early:
        evaluate(it)  # final evaluation at max_iters
    return {"iter": it, "best_val": best_val, "stopped_early": stopped_early}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the chess GPT.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--data", default="data/processed", help="dir with train.bin, val.bin, meta.json")
    p.add_argument("--out", default="runs/default", help="dir for checkpoints and logs")
    p.add_argument("--resume", action="store_true", help="resume from <out>/ckpt.pt")
    p.add_argument("--max-hours", type=float, help="stop and checkpoint after this many hours")
    p.add_argument("--iters-this-run", type=int, help="stop and checkpoint after this many iterations")
    p.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    max_seconds = args.max_hours * 3600 if args.max_hours is not None else None
    summary = train(cfg, args.data, args.out, resume=args.resume,
                    iters_this_run=args.iters_this_run, max_seconds=max_seconds, device=args.device)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
