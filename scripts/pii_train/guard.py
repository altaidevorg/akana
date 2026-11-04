"""Keeps long jobs within this laptop's limits: few CPU threads, and a pause whenever free memory
or commit runs low, the GPU runs hot or other work keeps the CPU busy.

The machine has 7.7 GB of RAM shared with other sessions and has cut power twice under combined CPU
and GPU load (Kernel-Power 41, no bugcheck), so training and inference loops call `wait()` every few
batches and start only when `headroom()` says there is room.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import time


def limit_threads(n: int = 4) -> None:
    """At most `n` CPU threads for numerical libraries; call before importing torch or numpy."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_gb() -> tuple[float, float]:
    """(free RAM, free commit) in GB."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 2**30, status.ullAvailPageFile / 2**30


_GPU_CACHE: list = [0.0, None]


def gpu(max_age_s: float = 5.0) -> tuple[int, int] | None:
    """(temperature in C, memory used in MiB), or None without an NVIDIA GPU. nvidia-smi takes
    ~0.4 s on Windows, so a reading is reused for `max_age_s`."""
    now = time.monotonic()
    if now - _GPU_CACHE[0] < max_age_s:
        return _GPU_CACHE[1]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        temp, used = (int(x) for x in out.split(","))
        value = (temp, used)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        value = None
    _GPU_CACHE[0], _GPU_CACHE[1] = now, value
    return value


def headroom(min_ram_gb: float = 1.0, min_commit_gb: float = 2.0, max_gpu_c: int = 75) -> tuple[bool, str]:
    ram, commit = memory_gb()
    g = gpu()
    state = f"free RAM {ram:.1f} GB, free commit {commit:.1f} GB" + (f", GPU {g[0]} C" if g else "")
    ok = ram >= min_ram_gb and commit >= min_commit_gb and (g is None or g[0] <= max_gpu_c)
    return ok, state


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]


_CPU_LAST: list = [None]


def cpu_busy() -> float | None:
    """Share of the machine's CPU time spent busy since the previous call (None on the first)."""
    idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    now = tuple((t.high << 32) | t.low for t in (idle, kernel, user))
    last, _CPU_LAST[0] = _CPU_LAST[0], now
    if last is None:
        return None
    total = (now[1] - last[1]) + (now[2] - last[2])  # kernel time includes idle time
    return 1 - (now[0] - last[0]) / total if total > 0 else None


def wait(min_ram_gb: float = 0.8, min_commit_gb: float = 1.5, max_gpu_c: int = 75, resume_gpu_c: int = 70,
         max_cpu: float = 0.85, poll_s: float = 10.0, log=print) -> None:
    """Blocks while memory is short, the GPU is above `max_gpu_c` (until it cools to `resume_gpu_c`)
    or other work keeps the CPU busier than `max_cpu` (measured since the previous call, so a check
    costs nothing). This GPU idles at 66-68 C (it shares its cooling with the CPU), so a resume
    point below that stalls a job for hours."""
    hot = False
    while True:
        ram, commit = memory_gb()
        g = gpu()
        temp = g[0] if g else 0
        hot = temp > (resume_gpu_c if hot else max_gpu_c)
        busy = cpu_busy()
        if ram >= min_ram_gb and commit >= min_commit_gb and not hot and (busy is None or busy <= max_cpu):
            return
        log(f"[guard] pausing: free RAM {ram:.1f} GB, free commit {commit:.1f} GB, GPU {temp} C"
            + (f", CPU {100 * busy:.0f}% busy" if busy is not None else ""))
        time.sleep(poll_s)


if __name__ == "__main__":
    print(headroom()[1])
