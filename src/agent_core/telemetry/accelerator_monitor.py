"""Optional accelerator-memory sampler.

NVIDIA collection uses NVML when ``nvidia-ml-py`` is installed.  Unsupported
platforms return explicit null values instead of pretending unified/process
memory is dedicated GPU memory.
"""

from __future__ import annotations

import os
from functools import lru_cache

from agent_core.telemetry.models import AcceleratorSnapshot


@lru_cache(maxsize=8)
def _nvml_handle(device_index: int):
    import pynvml  # type: ignore[import-not-found]

    pynvml.nvmlInit()
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(device_index)


def sample_accelerator(device_index: int = 0) -> AcceleratorSnapshot:
    try:
        pynvml, handle = _nvml_handle(device_index)
        device = pynvml.nvmlDeviceGetMemoryInfo(handle)
        process_used = None
        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            getter = getattr(pynvml, getter_name, None)
            if getter is None:
                continue
            for process in getter(handle):
                if int(process.pid) == os.getpid():
                    process_used = int(process.usedGpuMemory)
                    break
            break
        return AcceleratorSnapshot(
            backend="nvidia",
            device_index=device_index,
            process_used_bytes=process_used,
            device_used_bytes=int(device.used),
            device_total_bytes=int(device.total),
            supported=True,
        )
    except ImportError:
        return AcceleratorSnapshot(
            None, None, None, None, None, False, "nvidia-ml-py not installed"
        )
    except Exception as exc:
        return AcceleratorSnapshot(
            "nvidia", device_index, None, None, None, False, str(exc)
        )
