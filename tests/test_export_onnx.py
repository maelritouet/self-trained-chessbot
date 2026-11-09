"""Tests for scripts/export_onnx.py: best.pt -> model.onnx + meta.json for the web demo."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from chessgpt.model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("export_onnx", ROOT / "scripts" / "export_onnx.py")
export_onnx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_onnx)

ITOS = list("\n #+-./0123456789=BKNOQRabcdefghx")
MODEL_ARGS = dict(vocab_size=len(ITOS), block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A best.pt shaped like train.py's (default sdpa attention), exported once."""
    tmp = tmp_path_factory.mktemp("export")
    torch.manual_seed(0)
    model = GPT(GPTConfig(**MODEL_ARGS))
    ckpt = {"model": model.state_dict(), "model_args": MODEL_ARGS, "iter": 4500,
            "val_loss": 0.574, "itos": ITOS, "delimiter": "\n"}
    torch.save(ckpt, tmp / "best.pt")
    out = export_onnx.export_onnx(tmp / "best.pt", tmp / "web_model")
    return model.eval(), out


@pytest.mark.parametrize("B, T", [(1, 1), (1, 7), (3, 16)])
def test_onnx_logits_match_pytorch_for_any_batch_and_length(exported, B, T):
    model, out = exported
    idx = torch.randint(0, len(ITOS), (B, T), generator=torch.Generator().manual_seed(B * 100 + T))
    with torch.no_grad():
        expected = model(idx)[0].numpy()
    sess = ort.InferenceSession(str(out / "model.onnx"))
    (got,) = sess.run(["logits"], {"idx": idx.numpy()})
    assert got.shape == (B, T, len(ITOS))
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_onnx_io_names_are_the_web_contract(exported):
    _, out = exported
    sess = ort.InferenceSession(str(out / "model.onnx"))
    assert [i.name for i in sess.get_inputs()] == ["idx"]
    assert [o.name for o in sess.get_outputs()] == ["logits"]
    assert sess.get_inputs()[0].type == "tensor(int64)"


def test_meta_json_carries_everything_the_player_needs(exported):
    _, out = exported
    meta = json.loads((out / "meta.json").read_text())
    assert meta["itos"] == ITOS
    assert meta["vocab_size"] == len(ITOS)
    assert meta["delimiter"] == "\n"
    assert meta["block_size"] == 16
    assert meta["source"] == {"iter": 4500, "val_loss": 0.574}


def test_cli_writes_model_and_meta(exported, tmp_path):
    _, out = exported
    ckpt_path = out.parent / "best.pt"
    export_onnx.main([str(ckpt_path), "--out", str(tmp_path / "cli")])
    assert (tmp_path / "cli" / "model.onnx").exists() and (tmp_path / "cli" / "meta.json").exists()
