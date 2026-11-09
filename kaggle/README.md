# kaggle/

Training runs on Kaggle (private GPU notebook, T4: Kaggle retired the P100). Everything is
driven from the laptop with `.venv/bin/kaggle`; the token lives in `~/.kaggle/access_token`.

| File | What it does |
|------|--------------|
| `push_dataset.sh [data_dir] [msg]` | Uploads `train.bin`, `val.bin`, `meta.json` as the private dataset `<user>/chessgpt-data` (creates it, then adds versions). |
| `run.sh [--config …] [--name …] [--hours 11] [--no-smoke] [--resume]` | Builds the notebook for the current commit and starts it as `<user>/chessgpt-train`. Refuses dirty trees and unpushed commits. |
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
