# kaggle/

Training runs on Kaggle (private GPU notebook, T4: Kaggle retired the P100). Everything is
driven from the laptop with `.venv/bin/kaggle`; the token lives in `~/.kaggle/access_token`.

| File | What it does |
|------|--------------|
| `push_dataset.sh [data_dir] [msg]` | Uploads `train.bin`, `val.bin`, `meta.json` as the private dataset `<user>/chessgpt-data` (creates it, then adds versions; `KAGGLE_DATASET_SLUG=…` picks another slug). |
| `push_models.sh name=path …` | Uploads checkpoints as `<name>.pt` in the private dataset `<user>/chessgpt-models`, for `run.sh --init-from`. A new version replaces the file set: pass them all. |
| `run.sh [--config …] [--name …] [--hours 11] [--no-smoke] [--resume] [--dataset …] [--init-from x.pt]` | Builds the notebook for the current commit and starts it as `<user>/chessgpt-train`. Refuses dirty trees and unpushed commits. |
| `fetch_outputs.sh [name] [--no-ckpt]` | Downloads `runs/<name>/` (log, best.pt, ckpt.pt) into `runs/kaggle/`. |
| `train_kaggle.ipynb` | Thin template: clone at the pinned commit → install → find dataset → 30-step GPU smoke test → `python -m chessgpt.train --max-hours …`. |
| `build_kernel.py` | Fills the template's parameters cell and writes `kernel-metadata.json` (tested in `tests/test_build_kernel.py`). |
| `whoami.py` | Prints the Kaggle username (never the token). |

Typical loop:

```bash
kaggle/push_dataset.sh data/processed "first full dataset"   # once per dataset version
git push                                                     # run.sh needs HEAD on origin
kaggle/run.sh --name base --hours 11
.venv/bin/kaggle kernels logs maelrit/chessgpt-train --follow
kaggle/fetch_outputs.sh base
```

Resume (run longer than one 12 h session): `kaggle/run.sh --name base --resume` attaches
the previous version's output and continues from its `ckpt.pt`. Not yet tried on Kaggle.

Fine-tune (decision 012): train on another dataset, starting from a checkpoint's weights with a
fresh optimizer. The fine-tune data is its own dataset, attached *instead of* `chessgpt-data`
(the notebook wants exactly one `meta.json`); the checkpoints dataset has no `meta.json`.

```bash
KAGGLE_DATASET_SLUG=chessgpt-finetune kaggle/push_dataset.sh data/processed/finetune "long games"
kaggle/push_models.sh base=runs/kaggle/base-…/runs/base/best.pt large=runs/kaggle/large-…/runs/large/best.pt
kaggle/run.sh --config configs/finetune_large.yaml --name ft-large --dataset chessgpt-finetune --init-from large.pt
```

## What the first run showed (2026-11-09, `--name test --hours 0.5`, 20k-game sample)

- The session got a Tesla T4 (15 GB); the notebook cloned the pushed commit, installed it, and
  found the dataset at `/kaggle/input/datasets/maelrit/chessgpt-data` (newer Kaggle mount path;
  the notebook searches for `meta.json` instead of hard-coding it).
- The 30-step smoke test passed with fp16 on: val loss 3.695 → 2.088, no NaN.
- Throughput for the 10.7M-parameter `base.yaml`: ~100k tokens/s including evaluations
  (500 iterations of 32,768 tokens in ~163 s), so ~4B tokens per 11 h session.
- `--max-hours` stopped the run at iteration 5,501 and saved `ckpt.pt` before the session ended.
- The live log stream (`kernels logs --follow`) dropped once; `kernels status` kept working.
