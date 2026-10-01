"""Performance instrumentation utilities for PCSCS workflows.

The utilities in this module are deliberately analysis-agnostic: they record
runtime and resource telemetry around an existing computation without changing
that computation's semantics.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import os
import platform
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

try:
    import psutil
except ImportError:  # optional dependency used by performance workflows
    psutil = None


TELEMETRY_SCHEMA_VERSION = "1.0.0"


@dataclass
class ResourceSample:
    elapsed_s: float
    process_rss_bytes: Optional[int]
    process_cpu_percent: Optional[float]
    system_cpu_percent: Optional[float]
    system_memory_percent: Optional[float]
    gpu_utilization_percent: Optional[float]
    gpu_memory_used_bytes: Optional[int]
    gpu_memory_total_bytes: Optional[int]


class PerformanceMonitor:
    """Background resource sampler plus explicit phase/layer timing recorder."""

    def __init__(self, interval_s: float = 1.0):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = float(interval_s)
        self.started_perf: Optional[float] = None
        self.started_utc: Optional[str] = None
        self.finished_utc: Optional[str] = None
        self.samples: List[ResourceSample] = []
        self.phases: List[Dict[str, Any]] = []
        self.layers: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid()) if psutil else None

    def start(self) -> "PerformanceMonitor":
        if self._thread is not None:
            return self
        self.started_perf = time.perf_counter()
        self.started_utc = _utc_now()
        if self._process is not None:
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 2.0))
        self._sample_once()
        self.finished_utc = _utc_now()
        self._thread = None

    def _sample_loop(self) -> None:
        self._sample_once()
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        if self.started_perf is None:
            return
        elapsed = time.perf_counter() - self.started_perf
        rss = proc_cpu = sys_cpu = sys_mem = None
        if self._process is not None:
            try:
                rss = int(self._process.memory_info().rss)
                proc_cpu = float(self._process.cpu_percent(interval=None))
                sys_cpu = float(psutil.cpu_percent(interval=None))
                sys_mem = float(psutil.virtual_memory().percent)
            except Exception:
                pass
        gpu = _query_nvidia_smi()
        self.samples.append(
            ResourceSample(
                elapsed_s=float(elapsed),
                process_rss_bytes=rss,
                process_cpu_percent=proc_cpu,
                system_cpu_percent=sys_cpu,
                system_memory_percent=sys_mem,
                gpu_utilization_percent=gpu.get("utilization_percent"),
                gpu_memory_used_bytes=gpu.get("memory_used_bytes"),
                gpu_memory_total_bytes=gpu.get("memory_total_bytes"),
            )
        )

    @contextmanager
    def phase(self, name: str, **metadata: Any) -> Iterator[Dict[str, Any]]:
        """Time one named phase without modifying the wrapped computation."""
        started = time.perf_counter()
        record: Dict[str, Any] = {"name": name, "metadata": metadata}
        try:
            yield record
            record["ok"] = True
        except Exception as exc:
            record["ok"] = False
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["duration_s"] = float(time.perf_counter() - started)
            self.phases.append(record)

    @contextmanager
    def layer_phase(
        self,
        layer_name: str,
        phase_name: str,
        **metadata: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Time one layer-associated phase and append a normalized record."""
        started = time.perf_counter()
        record: Dict[str, Any] = {
            "layer": layer_name,
            "phase": phase_name,
            **metadata,
        }
        try:
            yield record
            record["ok"] = True
        except Exception as exc:
            record["ok"] = False
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["duration_s"] = float(time.perf_counter() - started)
            self.layers.append(record)

    def add_layer_metric(self, layer_name: str, **values: Any) -> None:
        self.layers.append({"layer": layer_name, "phase": "metric", **values})

    def to_dict(self) -> Dict[str, Any]:
        samples = [asdict(s) for s in self.samples]
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "sampling_interval_s": self.interval_s,
            "environment": runtime_environment(),
            "summary": summarize_samples(samples),
            "phases": self.phases,
            "layers": self.layers,
            "samples": samples,
        }


def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    def finite_values(key: str) -> List[float]:
        return [float(x[key]) for x in samples if x.get(key) is not None]

    def peak(key: str) -> Optional[float]:
        vals = finite_values(key)
        return max(vals) if vals else None

    def mean(key: str) -> Optional[float]:
        vals = finite_values(key)
        return sum(vals) / len(vals) if vals else None

    elapsed = finite_values("elapsed_s")
    return {
        "wall_clock_s": max(elapsed) if elapsed else None,
        "sample_count": len(samples),
        "peak_process_rss_bytes": _int_or_none(peak("process_rss_bytes")),
        "mean_process_cpu_percent": mean("process_cpu_percent"),
        "peak_process_cpu_percent": peak("process_cpu_percent"),
        "mean_system_cpu_percent": mean("system_cpu_percent"),
        "peak_system_cpu_percent": peak("system_cpu_percent"),
        "peak_system_memory_percent": peak("system_memory_percent"),
        "mean_gpu_utilization_percent": mean("gpu_utilization_percent"),
        "peak_gpu_utilization_percent": peak("gpu_utilization_percent"),
        "peak_gpu_memory_used_bytes": _int_or_none(peak("gpu_memory_used_bytes")),
        "gpu_memory_total_bytes": _int_or_none(peak("gpu_memory_total_bytes")),
    }


def runtime_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if psutil is not None:
        try:
            env["physical_cpu_count"] = psutil.cpu_count(logical=False)
            env["total_ram_bytes"] = int(psutil.virtual_memory().total)
        except Exception:
            pass
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_runtime"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_memory_bytes"] = int(props.total_memory)
    except Exception:
        pass
    return env


def torch_cuda_memory() -> Dict[str, Optional[int]]:
    """Return PyTorch CUDA allocator peaks when CUDA is active."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "current_allocated_bytes": None,
                "current_reserved_bytes": None,
            }
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "current_allocated_bytes": int(torch.cuda.memory_allocated()),
            "current_reserved_bytes": int(torch.cuda.memory_reserved()),
        }
    except Exception:
        return {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "current_allocated_bytes": None,
            "current_reserved_bytes": None,
        }


def reset_torch_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _query_nvidia_smi() -> Dict[str, Optional[float]]:
    if shutil.which("nvidia-smi") is None:
        return {
            "utilization_percent": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
        }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip().splitlines()[0]
        util, used_mib, total_mib = [float(v.strip()) for v in output.split(",")]
        mib = 1024 * 1024
        return {
            "utilization_percent": util,
            "memory_used_bytes": int(used_mib * mib),
            "memory_total_bytes": int(total_mib * mib),
        }
    except Exception:
        return {
            "utilization_percent": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
        }


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _int_or_none(value: Optional[float]) -> Optional[int]:
    return None if value is None else int(value)
