import csv
from pathlib import Path

from task.constants import METRICS_FIELDS
from task.metrics_io import ensure_metrics_schema


def test_ensure_metrics_schema_rotates_stale_header(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("timestamp,success,seed\n1,1,0\n", encoding="utf-8")
    bak = ensure_metrics_schema(str(path), METRICS_FIELDS)
    assert bak is not None
    assert Path(bak).is_file()
    assert not path.exists()


def test_ensure_metrics_schema_keeps_current_header(tmp_path):
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=METRICS_FIELDS)
        w.writeheader()
        w.writerow({k: "" for k in METRICS_FIELDS})
    assert ensure_metrics_schema(str(path), METRICS_FIELDS) is None
    assert path.is_file()
