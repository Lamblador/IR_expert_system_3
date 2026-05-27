"""Единое логирование пайплайна: консоль + файл, heartbeat для долгих операций."""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

_log_file: Path | None = None
_stage_name: str = ""


def configure_log(run_dir: Path | None = None, stage: str = "") -> None:
    global _log_file, _stage_name
    _stage_name = stage
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _log_file = run_dir / "pipeline.log"
    else:
        _log_file = None


def log(message: str, *, also_tqdm: bool = True) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    prefix = f"[ir-pipeline]"
    if _stage_name:
        prefix += f"[{_stage_name}]"
    line = f"{prefix} {ts} {message}"
    if also_tqdm:
        tqdm.write(line, file=sys.stderr)
    else:
        print(line, file=sys.stderr)
    if _log_file is not None:
        with _log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@contextmanager
def heartbeat(interval_s: float = 60.0, message: str = "still running...") -> Iterator[None]:
    """Периодически пишет в лог, пока блок кода выполняется."""
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval_s):
            log(message)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1.0)
