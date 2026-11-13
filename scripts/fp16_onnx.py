"""Halve an exported model's size: store its float32 weights as float16.

    python scripts/fp16_onnx.py web/models/base/model.onnx dist/fp16/base/model.onnx

Weight-only: every float32 initializer becomes float16, followed by a Cast back to float32,
so all arithmetic still runs in float32 and the graph's inputs and outputs are unchanged
(int64 "idx" in, float32 "logits" out). The only loss is rounding each weight to float16
(about 3 significant digits). A fully float16 graph would also halve the compute precision and
needs float16 kernels that the browser's WASM backend doesn't have for every op.

Small tensors (fewer than --min-elements values: biases, LayerNorm scales) stay float32: they
weigh almost nothing and keeping them exact is free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def to_fp16_weights(src: str | Path, dst: str | Path, min_elements: int = 1024) -> tuple[int, int]:
    """Rewrite src into dst with large float32 initializers stored as float16.

    Returns (number of tensors converted, number of values converted)."""
    model = onnx.load(str(src))
    graph = model.graph
    casts, kept = [], []
    n_tensors = n_values = 0
    for init in graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT or np.prod(init.dims, dtype=np.int64) < min_elements:
            kept.append(init)
            continue
        half_name = init.name + "__fp16"
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), half_name)
        kept.append(half)
        # The Cast writes the original name, so every consumer is unchanged.
        casts.append(helper.make_node("Cast", [half_name], [init.name], to=onnx.TensorProto.FLOAT,
                                      name=f"cast_{init.name}"))
        n_tensors += 1
        n_values += half.ByteSize() // 2
    kept = _dedupe(graph, kept)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    nodes = casts + list(graph.node)  # casts first keeps the graph topologically sorted
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return n_tensors, n_values


def _dedupe(graph, inits: list) -> list:
    """Keep one copy of byte-identical initializers and point every consumer at it.

    Each attention block carries its own causal mask buffer (1024 x 1024 bools, 1 MB), all equal."""
    first: dict[tuple, str] = {}
    rename: dict[str, str] = {}
    out = []
    for init in inits:
        key = (init.data_type, tuple(init.dims), numpy_helper.to_array(init).tobytes())
        if key in first:
            rename[init.name] = first[key]
        else:
            first[key] = init.name
            out.append(init)
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="float32 model.onnx (from scripts/export_onnx.py)")
    p.add_argument("dst", help="where to write the float16-weight copy")
    p.add_argument("--min-elements", type=int, default=1024, help="smaller tensors stay float32")
    a = p.parse_args(argv)
    n, _ = to_fp16_weights(a.src, a.dst, a.min_elements)
    mb = lambda path: Path(path).stat().st_size / 1e6
    print(f"{a.dst}: {n} tensors to float16, {mb(a.src):.1f} MB -> {mb(a.dst):.1f} MB")


if __name__ == "__main__":
    main()
