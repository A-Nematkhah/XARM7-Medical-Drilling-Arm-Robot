import json

from task.log import RunLog


def test_runlog_writes_jsonl(tmp_path):
    path = tmp_path / "run.jsonl"
    log = RunLog(path, sim_t=lambda: 1.25, run_meta={"seed": 3})
    log.event("5/8 locate board", "end", hole_err_mm=8.4)
    log.close(success=True)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    recs = [json.loads(x) for x in lines]
    assert recs[0]["phase"] == "run" and recs[0]["event"] == "start"
    assert recs[0]["seed"] == 3
    mid = next(r for r in recs if r["phase"].startswith("5/"))
    assert mid["hole_err_mm"] == 8.4
    assert recs[-1]["event"] == "end"
    assert recs[-1]["success"] is True
