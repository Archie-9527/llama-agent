"""可选的加速器内存采样器。

安装 ``nvidia-ml-py`` 后，通过 NVML 采集 NVIDIA 指标。不受支持的平台会返回
明确的空值，而不会把统一内存或进程内存误报为 GPU 专用显存。
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
