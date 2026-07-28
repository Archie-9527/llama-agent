"""用作 Benchmark 原始数据的线程安全追加写文件。"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any


class TelemetrySink:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        with self._lock:
            with (self.output_dir / filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str))
                handle.write("\n")

    def append_csv(self, filename: str, record: dict[str, Any]) -> None:
        path = self.output_dir / filename
        with self._lock:
            exists = path.exists() and path.stat().st_size > 0
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(record)
