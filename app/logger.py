import json
import pathlib
import time

LOG_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "turns.jsonl"


def log_turn(record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    record["ts"] = time.time()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")