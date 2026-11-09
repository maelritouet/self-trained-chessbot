"""Export a best.pt checkpoint to ONNX for the browser demo.

    python scripts/export_onnx.py runs/<run>/best.pt --out web/models/base

Writes <out>/model.onnx (input "idx": int64 (batch, seq); output "logits": float32
(batch, seq, vocab)) and <out>/meta.json (vocab, delimiter, block_size, provenance),
which is everything web/js/player.js needs.

The model is rebuilt with attn_impl="manual": the explicit masked-softmax attention
exports to plain ONNX ops that every runtime supports, and it shares parameters with the
fused path used in training, so the checkpoint loads unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chessgpt.model import GPT, GPTConfig


class _LogitsOnly(torch.nn.Module):
    """ONNX graphs need a fixed set of outputs: drop the (None) loss."""

    def __init__(self, model: GPT):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)[0]


def export_onnx(ckpt_path: str | Path, out_dir: str | Path, opset: int = 18) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = {**ckpt["model_args"], "attn_impl": "manual", "dropout": 0.0}
    model = GPT(GPTConfig(**args))
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    block = args["block_size"]
    dummy = torch.zeros((2, min(8, block)), dtype=torch.long)
    batch = torch.export.Dim("batch", min=1)
    seq = torch.export.Dim("seq", min=1, max=block)
    torch.onnx.export(
        _LogitsOnly(model).eval(), (dummy,), str(out_dir / "model.onnx"),
        input_names=["idx"], output_names=["logits"],
        dynamic_shapes={"idx": {0: batch, 1: seq}},
        opset_version=opset, dynamo=True, external_data=False,
    )

    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "vocab_size": args["vocab_size"],
        "itos": ckpt["itos"],
        "delimiter": ckpt.get("delimiter", "\n"),
        "block_size": block,
        "n_params": n_params,
        "source": {"iter": ckpt.get("iter"), "val_loss": ckpt.get("val_loss")},
        "opset": opset,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", help="best.pt written by chessgpt.train")
    p.add_argument("--out", default="web/models/base")
    p.add_argument("--opset", type=int, default=18)
    a = p.parse_args(argv)
    out = export_onnx(a.ckpt, a.out, a.opset)
    size = (out / "model.onnx").stat().st_size / 1e6
    print(f"{out / 'model.onnx'} ({size:.1f} MB), {out / 'meta.json'}")


if __name__ == "__main__":
    main()
