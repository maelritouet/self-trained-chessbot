"""Training entry point:  python -m chessgpt.train --config configs/base.yaml --data data/processed --out runs/x

TODO: the training loop. Provided: CLI, config loading,
and get_batch (random windows from the memmapped .bin files).

TODO (the loop):
  [ ] seed everything (torch, numpy, random) from train.seed
  [ ] read meta.json -> vocab_size; build GPTConfig + GPT; move to device
  [ ] AdamW (weight decay on 2D weights only, not on biases / LayerNorm)
  [ ] LR schedule: linear warmup for warmup_iters, then cosine decay to min_lr at max_iters
  [ ] fp16 on GPU: torch.autocast(device_type="cuda", dtype=torch.float16) + torch.amp.GradScaler
  [ ] gradient accumulation (grad_accum_steps) + gradient clipping (grad_clip; unscale first!)
  [ ] every eval_interval: estimate train & val loss over eval_iters batches (model.eval(), no_grad)
  [ ] CSV log: iter, lr, train_loss, val_loss, tokens_seen, seconds
  [ ] checkpoint every ckpt_interval: model, optimizer, scaler, iter, config, best_val, RNG states
  [ ] --resume: load the checkpoint and continue exactly where it stopped (Kaggle runs cap at 12 h)

Questions to be able to answer:
  - Why warmup? What happens to Adam's early updates without it?
  - Why does GradScaler exist, and why must gradients be unscaled before clipping?
  - What has to be saved so a resumed run is indistinguishable from an uninterrupted one?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def get_batch(
    data_dir: str | Path, split: str, batch_size: int, block_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows from {split}.bin.

    Returns:
        x: (batch_size, block_size) int64, tokens [i, i+block_size)
        y: (batch_size, block_size) int64, the same windows shifted by one: [i+1, i+block_size+1)

    Windows may span several games; the delimiter token marks where a new one begins.
    The memmap is reopened on every call (as in nanoGPT): a long-lived np.memmap keeps
    growing in memory as pages are touched.
    """
    data = np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint8, mode="r")
    if len(data) <= block_size + 1:
        raise ValueError(f"{split}.bin has {len(data)} tokens, need > block_size + 1 = {block_size + 1}")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix.tolist()])
    if device.startswith("cuda"):
        # pinned memory + non_blocking lets the copy overlap with GPU compute
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the chess GPT.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--data", default="data/processed", help="dir with train.bin, val.bin, meta.json")
    p.add_argument("--out", default="runs/default", help="dir for checkpoints and logs")
    p.add_argument("--resume", action="store_true", help="resume from <out>/ckpt.pt")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    raise NotImplementedError("training loop not implemented yet, see the TODO list above")


if __name__ == "__main__":
    main()
