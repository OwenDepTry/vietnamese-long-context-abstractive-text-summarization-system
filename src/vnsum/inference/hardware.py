"""Thông tin phần cứng thật và đo bộ nhớ, chỉ dùng thư viện chuẩn (không thêm psutil / pynvml).

* RAM đỉnh của process: Windows ``PeakWorkingSetSize`` (psapi), Linux/macOS ``getrusage().ru_maxrss``.
* VRAM: lấy mẫu ``nvidia-smi --query-gpu=memory.used`` mỗi 100 ms trong lúc chạy, trừ mức nền đo
  trước khi nạp model. Đây là bộ nhớ của CẢ GPU (gồm mọi process khác), nên chỉ chính xác khi không
  có chương trình khác dùng GPU trong lúc đo; với PyTorch có thêm ``torch.cuda.max_memory_allocated``.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

MB = 1024 * 1024


# --------------------------------------------------------------------------- RAM
def peak_rss_mb() -> float | None:
    """RAM đỉnh (working set / max RSS) của process hiện tại, MB."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return counters.PeakWorkingSetSize / MB
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / MB if sys.platform == "darwin" else peak / 1024  # macOS: byte, Linux: KB
    except Exception:  # pragma: no cover - phụ thuộc hệ điều hành
        return None


# --------------------------------------------------------------------------- GPU
def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def gpu_info() -> list[dict[str, str]]:
    if not nvidia_smi_available():
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory_total_mb": parts[1], "driver": parts[2]})
    return gpus


class GpuMemorySampler:
    """Lấy mẫu memory.used của GPU ``index`` bằng ``nvidia-smi -lms`` trong luồng nền."""

    def __init__(self, index: int = 0, interval_ms: int = 100) -> None:
        self.index, self.interval_ms = index, interval_ms
        self.samples: list[float] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def read_once(self) -> float | None:
        try:
            out = subprocess.run(["nvidia-smi", f"--id={self.index}", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
            return float(out.strip().splitlines()[0])
        except Exception:
            return None

    def start(self) -> None:
        if not nvidia_smi_available():
            return
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--id={self.index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             f"-lms={self.interval_ms}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def _read() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                try:
                    self.samples.append(float(line.strip()))
                except ValueError:
                    continue

        self._thread = threading.Thread(target=_read, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return max(self.samples) if self.samples else None


# --------------------------------------------------------------------------- CPU / hệ thống
def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return platform.processor() or "không xác định"


def total_ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1024 ** 3
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 1024 ** 3
        m = re.search(r"^MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        return int(m.group(1)) / 1024 ** 2 if m else None
    except Exception:
        return None


def library_versions() -> dict[str, str]:
    from importlib import metadata

    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "optimum", "optimum-onnx", "onnxruntime", "onnxruntime-gpu", "onnx"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def system_info() -> dict[str, Any]:
    ram = total_ram_gb()
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram_gb": round(ram, 1) if ram else None,
        "gpus": gpu_info(),
        "libraries": library_versions(),
    }
    try:
        import onnxruntime as ort

        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        info["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def dir_size_mb(path: str | Path, patterns: tuple[str, ...]) -> float:
    """Tổng kích thước các file trọng số trong ``path`` (không đệ quy), MB."""
    path = Path(path)
    seen: set[Path] = set()
    for pat in patterns:
        seen.update(p for p in path.glob(pat) if p.is_file())
    return sum(p.stat().st_size for p in seen) / MB


PYTORCH_WEIGHT_PATTERNS = ("*.safetensors", "*.bin")
ONNX_WEIGHT_PATTERNS = ("*.onnx", "*.onnx_data", "*.onnx.data")
