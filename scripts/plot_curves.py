"""Plot loss curves from one or more training log.csv files.

    python scripts/plot_curves.py runs/kaggle/test-*/runs/test/log.csv
    python scripts/plot_curves.py runA/log.csv runB/log.csv --labels base wide --out plots/compare.png

Two stacked panels sharing the x-axis (tokens seen): loss (val solid, train dashed, one
color per run, best val point marked) and learning rate. The LR gets its own panel rather
than a second y-axis: two scales on one plot invite false comparisons.
Default output: loss_curves.png next to the first log.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed (works on Kaggle and headless)
import matplotlib.pyplot as plt  # noqa: E402

INT_COLUMNS = ("iter", "tokens_seen")
FLOAT_COLUMNS = ("lr", "train_loss", "val_loss", "elapsed_s")

# Categorical slots in fixed order (validated reference palette, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load_log(path: str | Path) -> dict[str, list]:
    """Read a train.py log.csv into typed columns. Raises ValueError if it has no rows."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no logged evaluations yet")
    log: dict[str, list] = {}
    for col in INT_COLUMNS:
        log[col] = [int(r[col]) for r in rows]
    for col in FLOAT_COLUMNS:
        log[col] = [float(r[col]) for r in rows]
    return log


def best_val(log: dict[str, list]) -> dict:
    """The evaluation with the lowest val loss (what best.pt holds)."""
    i = min(range(len(log["val_loss"])), key=log["val_loss"].__getitem__)
    return {"iter": log["iter"][i], "tokens_seen": log["tokens_seen"][i], "val_loss": log["val_loss"][i]}


def _tokens_formatter(x, _pos) -> str:
    if x >= 1e9:
        return f"{x / 1e9:g}B"
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    return f"{x:g}"


def plot_runs(logs: dict[str, dict[str, list]], out_path: str | Path) -> Path:
    """Plot every run in `logs` ({label: log}) into one PNG. Returns the output path."""
    if len(logs) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} runs per plot (got {len(logs)}); split into several plots")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE)
    for ax in (ax_loss, ax_lr):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for color, (label, log) in zip(SERIES, logs.items()):
        x = log["tokens_seen"]
        ax_loss.plot(x, log["val_loss"], color=color, linewidth=2, label=f"{label} val")
        ax_loss.plot(x, log["train_loss"], color=color, linewidth=2, linestyle=(0, (4, 3)),
                     alpha=0.8, label=f"{label} train")
        b = best_val(log)
        ax_loss.plot(b["tokens_seen"], b["val_loss"], "o", markersize=8, color=color,
                     markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax_loss.annotate(f"best val {b['val_loss']:.3f}\n(iter {b['iter']:,})",
                         (b["tokens_seen"], b["val_loss"]), xytext=(8, 10),
                         textcoords="offset points", fontsize=9, color=INK_2)
        ax_lr.plot(x, log["lr"], color=color, linewidth=2, label=label)

    # Log scale: the first steps drop loss ~4x, which on a linear axis squashes the late
    # train/val gap (the overfitting signal) into a few pixels.
    ax_loss.set_yscale("log")
    ax_loss.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax_loss.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax_loss.set_ylabel("loss (nats / character, log scale)")
    ax_loss.set_title("Training curves", loc="left", color=INK, fontsize=12)
    ax_loss.legend(frameon=False, ncols=min(2 * len(logs), 4), loc="upper right")
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("tokens seen")
    ax_lr.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_tokens_formatter))
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="log.csv files written by chessgpt.train")
    p.add_argument("--labels", nargs="+", help="one label per log (default: the run directory name)")
    p.add_argument("--out", help="output PNG (default: loss_curves.png next to the first log)")
    a = p.parse_args(argv)
    if a.labels and len(a.labels) != len(a.logs):
        p.error(f"{len(a.labels)} labels for {len(a.logs)} logs")
    labels = a.labels or [Path(path).parent.name for path in a.logs]
    out = Path(a.out) if a.out else Path(a.logs[0]).parent / "loss_curves.png"
    logs = {label: load_log(path) for label, path in zip(labels, a.logs)}
    print(plot_runs(logs, out))
    for label, log in logs.items():
        b = best_val(log)
        print(f"{label}: best val {b['val_loss']:.4f} at iter {b['iter']:,} ({b['tokens_seen']:,} tokens)")


if __name__ == "__main__":
    main()
