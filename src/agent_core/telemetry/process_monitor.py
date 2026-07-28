"""无需强制第三方依赖的可移植进程内存采样器。"""

from __future__ import annotations

import os
import platform
import resource
from datetime import datetime, timezone
from time import monotonic_ns

from agent_core.telemetry.models import ProcessSnapshot


def _from_psutil() -> tuple[int | None, int | None, int | None]:
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process(os.getpid())
        basic = process.memory_info()
        try:
            full = process.memory_full_info()
            uss = getattr(full, "uss", None)
        except (AttributeError, OSError, psutil.Error):
            uss = None
        return int(basic.rss), int(basic.vms), int(uss) if uss is not None else None
    except (ImportError, OSError):
        return _from_stdlib()


def _from_stdlib() -> tuple[int | None, int | None, int | None]:
    rss: int | None = None
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS 返回字节，Linux 和大多数 BSD 变体返回 KiB。
        rss = int(raw if platform.system() == "Darwin" else raw * 1024)
    except (OSError, ValueError):
        pass

    vms: int | None = None
    if platform.system() == "Linux":
        try:
            for line in open(f"/proc/{os.getpid()}/status", encoding="utf-8"):
                if line.startswith("VmSize:"):
                    vms = int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
    return rss, vms, None


def sample_process() -> ProcessSnapshot:
    rss, vms, uss = _from_psutil()
    return ProcessSnapshot(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        monotonic_ns=monotonic_ns(),
        rss_bytes=rss,
        vms_bytes=vms,
        uss_bytes=uss,
    )
