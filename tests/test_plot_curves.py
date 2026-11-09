"""Tests for scripts/plot_curves.py: reading log.csv files and plotting loss curves."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("plot_curves", ROOT / "scripts" / "plot_curves.py")
plot_curves = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_curves)

HEADER = "iter,lr,train_loss,val_loss,tokens_seen,elapsed_s\n"
LOG = HEADER + (
    "0,1.2e-06,3.698036,3.697679,0,17.2\n"
    "500,0.0006,1.069185,1.073840,16384000,182.6\n"
    "1000,0.000599124,0.783141,0.783563,32768000,344.9\n"
    "1500,0.000596504,0.674802,0.790000,49152000,507.7\n"
)


@pytest.fixture
def log_path(tmp_path):
    p = tmp_path / "run" / "log.csv"
    p.parent.mkdir()
    p.write_text(LOG)
    return p


def test_load_log_returns_typed_columns(log_path):
    log = plot_curves.load_log(log_path)
    assert log["iter"] == [0, 500, 1000, 1500]
    assert log["tokens_seen"] == [0, 16384000, 32768000, 49152000]
    assert log["val_loss"] == pytest.approx([3.697679, 1.073840, 0.783563, 0.790000])
    assert log["lr"][1] == pytest.approx(6e-4)
    assert all(isinstance(v, int) for v in log["iter"] + log["tokens_seen"])


def test_load_log_rejects_a_log_without_rows(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text(HEADER)
    with pytest.raises(ValueError):
        plot_curves.load_log(p)


def test_best_val_is_the_minimum_not_the_last_row(log_path):
    best = plot_curves.best_val(plot_curves.load_log(log_path))
    assert best == {"iter": 1000, "tokens_seen": 32768000, "val_loss": pytest.approx(0.783563)}


def test_plot_writes_a_png_for_several_runs(log_path, tmp_path):
    other = tmp_path / "other.csv"
    other.write_text(LOG)
    out = tmp_path / "plots" / "curves.png"
    logs = {"a": plot_curves.load_log(log_path), "b": plot_curves.load_log(other)}
    assert plot_curves.plot_runs(logs, out) == out
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_default_output_sits_next_to_the_first_log(log_path):
    plot_curves.main([str(log_path)])
    assert (log_path.parent / "loss_curves.png").exists()


def test_cli_labels_must_match_number_of_logs(log_path, tmp_path):
    with pytest.raises(SystemExit):
        plot_curves.main([str(log_path), "--labels", "a", "b", "--out", str(tmp_path / "x.png")])
