"""CSV metrics schema: rotate a stale header so DictReader stays aligned."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone


def ensure_metrics_schema(path, fields):
    """If `path` exists with a different header, archive it and start a new file.

    Returns the archive path (or None if no rotation).
    """
    fields = list(fields)
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    if header == fields:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root, ext = os.path.splitext(path)
    bak = f"{root}_legacy_{ts}{ext or '.csv'}"
    os.replace(path, bak)
    return bak
