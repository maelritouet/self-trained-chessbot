"""Build a pushable Kaggle kernel folder from the notebook template.

Writes <out_dir>/train_kaggle.ipynb (template with its "parameters" cell rewritten)
and <out_dir>/kernel-metadata.json, ready for `kaggle kernels push -p <out_dir>`.
Called by kaggle/run.sh; usable on its own:

    python kaggle/build_kernel.py --user maelrit --repo-url https://github.com/u/chessgpt.git \
        --git-ref $(git rev-parse HEAD) --run-name base --out kaggle/build/kernel
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

PARAM_KEYS = ("REPO_URL", "GIT_REF", "CONFIG", "RUN_NAME", "MAX_HOURS", "SMOKE_TEST", "RESUME", "INIT_FROM")
# Fine-tuning (INIT_FROM = a file name such as "large.pt") reads its starting weights from this
# private dataset of best.pt files (kaggle/push_models.sh). It has no meta.json, so the
# notebook's "exactly one meta.json under /kaggle/input" lookup still finds the training data.
MODELS_DATASET = "chessgpt-models"
KAGGLE_SESSION_HOURS = 12.0
# Margin under the 12 h hard limit: setup + smoke test + the final evaluation and
# checkpoint after --max-hours triggers must all fit before Kaggle kills the session.
MAX_TRAIN_HOURS = 11.5
CODE_FILE = "train_kaggle.ipynb"


def _param_source(params: dict) -> list[str]:
    lines = ["# Parameters: rewritten by kaggle/run.sh (kaggle/build_kernel.py)\n"]
    # repr() gives valid Python literals for str/float/bool, including quotes and escapes.
    lines += [f"{k} = {params[k]!r}\n" for k in PARAM_KEYS]
    return lines


def build_kernel(
    template: str | Path, out_dir: str | Path, params: dict, *, user: str,
    slug: str = "chessgpt-train", dataset: str = "chessgpt-data", accelerator: str = "NvidiaTeslaT4",
) -> Path:
    """Write the filled notebook + kernel-metadata.json into out_dir. Returns out_dir."""
    if set(params) != set(PARAM_KEYS):
        missing, extra = set(PARAM_KEYS) - set(params), set(params) - set(PARAM_KEYS)
        raise ValueError(f"parameters must be exactly {PARAM_KEYS}; missing {missing}, unknown {extra}")
    if not 0 < params["MAX_HOURS"] <= MAX_TRAIN_HOURS:
        raise ValueError(f"MAX_HOURS must be in (0, {MAX_TRAIN_HOURS}] to finish inside "
                         f"Kaggle's {KAGGLE_SESSION_HOURS:g} h session, got {params['MAX_HOURS']}")
    if params["RESUME"] and params["INIT_FROM"]:
        raise ValueError("RESUME and INIT_FROM are exclusive (as in chessgpt.train): "
                         "resume a fine-tune with RESUME alone")

    nb = json.loads(Path(template).read_text())
    nb = copy.deepcopy(nb)
    param_cells = [c for c in nb["cells"] if "parameters" in c.get("metadata", {}).get("tags", [])]
    if len(param_cells) != 1:
        raise ValueError(f"template needs exactly one cell tagged 'parameters', found {len(param_cells)}")
    param_cells[0]["source"] = _param_source(params)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CODE_FILE).write_text(json.dumps(nb, indent=1) + "\n")

    metadata = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,          # needed for git clone + pip install
        "machine_shape": accelerator,     # P100 is retired on Kaggle; T4 is the default GPU
        "dataset_sources": [f"{user}/{dataset}"] + ([f"{user}/{MODELS_DATASET}"] if params["INIT_FROM"] else []),
        # Resume: attach this kernel's previous version output (contains runs/<RUN_NAME>/ckpt.pt).
        "kernel_sources": [f"{user}/{slug}"] if params["RESUME"] else [],
        "competition_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return out_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True, help="Kaggle username")
    p.add_argument("--repo-url", required=True)
    p.add_argument("--git-ref", required=True, help="commit hash to check out on Kaggle")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--run-name", default="base")
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--no-smoke", action="store_true", help="skip the 30-step GPU smoke test")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dataset", default="chessgpt-data", help="private Kaggle dataset with the .bin files")
    p.add_argument("--init-from", default="", help=f"fine-tune from this file of {MODELS_DATASET}, e.g. large.pt")
    p.add_argument("--template", default=str(Path(__file__).with_name(CODE_FILE)))
    p.add_argument("--out", default="kaggle/build/kernel")
    a = p.parse_args(argv)
    params = dict(REPO_URL=a.repo_url, GIT_REF=a.git_ref, CONFIG=a.config, RUN_NAME=a.run_name,
                  MAX_HOURS=a.max_hours, SMOKE_TEST=not a.no_smoke, RESUME=a.resume,
                  INIT_FROM=a.init_from)
    print(build_kernel(a.template, a.out, params, user=a.user, dataset=a.dataset))


if __name__ == "__main__":
    main()
