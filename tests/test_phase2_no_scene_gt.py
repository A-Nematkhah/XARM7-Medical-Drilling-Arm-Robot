"""Phase 2: live task code must not read MuJoCo BOARD_POS / BOARD_HALF."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "task"


def test_task_package_has_no_scene_board_constants():
    hits = []
    for path in TASK.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in ("BOARD_POS", "BOARD_HALF", "import scene", "from scene"):
            if token in text:
                hits.append(f"{path.relative_to(ROOT)}: {token}")
    assert hits == []
