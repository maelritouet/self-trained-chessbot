"""Tests for kaggle/build_kernel.py: filling the notebook's parameters cell and
writing kernel-metadata.json for `kaggle kernels push`.

kaggle/ is not a package (and `import kaggle` is the Kaggle CLI), so the module is
loaded from its file path.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "kaggle" / "train_kaggle.ipynb"

_spec = importlib.util.spec_from_file_location("build_kernel", ROOT / "kaggle" / "build_kernel.py")
build_kernel_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_kernel_mod)
build_kernel = build_kernel_mod.build_kernel

PARAMS = dict(
    REPO_URL="https://github.com/someone/chessgpt.git",
    GIT_REF="0123456789abcdef0123456789abcdef01234567",
    CONFIG="configs/base.yaml",
    RUN_NAME="base-v1",
    MAX_HOURS=11.0,
    SMOKE_TEST=True,
    RESUME=False,
)


def params(**overrides):
    return {**PARAMS, **overrides}


def built_notebook(out_dir):
    return json.loads((out_dir / "train_kaggle.ipynb").read_text())


def param_cell_values(nb) -> dict:
    """Execute the parameters cell and return the variables it defines."""
    cells = [c for c in nb["cells"] if "parameters" in c["metadata"].get("tags", [])]
    assert len(cells) == 1
    ns: dict = {}
    exec("".join(cells[0]["source"]), ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}


def test_parameters_cell_holds_the_given_values(tmp_path):
    build_kernel(TEMPLATE, tmp_path, params(), user="someone")
    assert param_cell_values(built_notebook(tmp_path)) == PARAMS


def test_string_values_with_quotes_survive(tmp_path):
    build_kernel(TEMPLATE, tmp_path, params(RUN_NAME='it\'s "quoted"'), user="someone")
    assert param_cell_values(built_notebook(tmp_path))["RUN_NAME"] == 'it\'s "quoted"'


def test_other_cells_are_unchanged(tmp_path):
    build_kernel(TEMPLATE, tmp_path, params(), user="someone")
    template = json.loads(TEMPLATE.read_text())
    built = built_notebook(tmp_path)
    assert len(built["cells"]) == len(template["cells"])
    for t, b in zip(template["cells"], built["cells"]):
        if "parameters" not in t["metadata"].get("tags", []):
            assert t == b


def test_template_file_is_not_modified(tmp_path):
    before = TEMPLATE.read_bytes()
    build_kernel(TEMPLATE, tmp_path, params(), user="someone")
    assert TEMPLATE.read_bytes() == before


def test_metadata_requests_private_gpu_notebook_with_dataset(tmp_path):
    build_kernel(TEMPLATE, tmp_path, params(), user="someone")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/chessgpt-train"
    assert meta["code_file"] == "train_kaggle.ipynb"
    assert meta["kernel_type"] == "notebook"
    assert meta["language"] == "python"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == ["someone/chessgpt-data"]
    assert meta["kernel_sources"] == []


def test_resume_attaches_previous_output_of_the_same_kernel(tmp_path):
    build_kernel(TEMPLATE, tmp_path, params(RESUME=True), user="someone")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["someone/chessgpt-train"]


def test_unknown_or_missing_parameter_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_kernel(TEMPLATE, tmp_path, params(TYPO=1), user="someone")
    missing = params()
    del missing["GIT_REF"]
    with pytest.raises(ValueError):
        build_kernel(TEMPLATE, tmp_path, missing, user="someone")


def test_max_hours_must_leave_margin_under_kaggle_12h_limit(tmp_path):
    with pytest.raises(ValueError):
        build_kernel(TEMPLATE, tmp_path, params(MAX_HOURS=11.75), user="someone")


def test_template_without_parameters_cell_is_rejected(tmp_path):
    nb = json.loads(TEMPLATE.read_text())
    for c in nb["cells"]:
        c["metadata"].pop("tags", None)
    bad = tmp_path / "bad.ipynb"
    bad.write_text(json.dumps(nb))
    with pytest.raises(ValueError):
        build_kernel(bad, tmp_path / "out", params(), user="someone")
