from pathlib import Path
import json
import time
import uuid

def make_run_dir(base="runs"):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base) / ts / run_id
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir, run_id

def save_summary(run_dir, summary: dict):
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
