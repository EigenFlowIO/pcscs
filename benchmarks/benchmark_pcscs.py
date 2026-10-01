#!/usr/bin/env python3
"""Controlled PCSCS scaling benchmark.

This benchmark isolates PCSCS from neural-network feature extraction. For each
configured sample size it generates a deterministic fixed feature matrix,
computes the cosine-similarity matrix with the public package utility, and runs
PCSCS on that matrix. The first run is a warm-up; subsequent runs are measured.

Outputs are JSON and CSV with complete provenance and resource telemetry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np

from pcscs import PCSCS, compute_similarity_matrix
from pcscs.performance import PerformanceMonitor, runtime_environment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")
DEFAULT_OUTPUT = ROOT / "benchmark_results"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_info() -> Dict[str, Any]:
    def capture(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    return {
        "commit": capture("rev-parse", "HEAD"),
        "branch": capture("rev-parse", "--abbrev-ref", "HEAD"),
        "remote_origin": capture("remote", "get-url", "origin"),
        "dirty": bool(capture("status", "--porcelain")),
    }


def fixed_features(seed: int, n: int, d: int) -> np.ndarray:
    # A per-condition seed prevents sample-count order from changing the matrix.
    rng = np.random.default_rng(seed + n * 1000003 + d)
    features = rng.standard_normal((n, d), dtype=np.float32)
    # Add weak deterministic group structure so threshold dynamics are nontrivial.
    group_count = min(5, max(2, n // 25))
    offsets = rng.standard_normal((group_count, d), dtype=np.float32) * 0.20
    groups = np.arange(n) % group_count
    features += offsets[groups]
    return features.astype(np.float32, copy=False)


def one_run(
    features: np.ndarray,
    n_steps: int,
    enable_tracking: bool,
    telemetry_interval_s: float,
) -> Dict[str, Any]:
    monitor = PerformanceMonitor(interval_s=telemetry_interval_s).start()
    started = time.perf_counter()
    try:
        with monitor.phase("similarity"):
            similarity = compute_similarity_matrix(features)
        with monitor.phase("pcscs"):
            result = PCSCS(enable_tracking=enable_tracking).analyze_layer(
                similarity,
                layer_name="benchmark",
                n_steps=n_steps,
            )
        total_s = time.perf_counter() - started
    finally:
        monitor.stop()

    phase_times = {row["name"]: row["duration_s"] for row in monitor.phases}
    return {
        "total_s": float(total_s),
        "similarity_s": float(phase_times["similarity"]),
        "pcscs_s": float(phase_times["pcscs"]),
        "convergence_threshold": float(result.convergence_threshold),
        "critical_threshold": float(result.critical_threshold),
        "critical_rate": float(result.critical_rate),
        "fit_successful": bool(result.fit_successful),
        "telemetry_summary": monitor.to_dict()["summary"],
    }


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def run_benchmark(config_path: Path, output_dir: Path, sample_sizes: List[int] | None = None) -> Dict[str, Any]:
    cfg = load_json(config_path)
    seed = int(cfg["seed"])
    sizes = list(sample_sizes or cfg["sample_sizes"])
    d = int(cfg["feature_dimension"])
    n_steps = int(cfg["n_steps"])
    tracking = bool(cfg["enable_tracking"])
    warmups = int(cfg["warmup_runs"])
    repeats = int(cfg["measured_repeats"])
    interval = float(cfg["telemetry_interval_s"])

    if not sizes or any(int(n) < 2 for n in sizes):
        raise ValueError("sample sizes must all be >= 2")
    if d < 1 or n_steps < 2 or repeats < 1 or warmups < 0:
        raise ValueError("invalid benchmark configuration")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: List[Dict[str, Any]] = []
    conditions: Dict[str, Any] = {}
    max_n = max(int(n) for n in sizes)
    base_features = fixed_features(seed, max_n, d)

    for n in sizes:
        n = int(n)
        # Nested subsets hold the representation-generating process fixed as n grows.
        features = base_features[:n].copy()
        feature_sha = hashlib.sha256(features.tobytes(order="C")).hexdigest()
        print(f"benchmark n={n}, d={d}, steps={n_steps}, tracking={tracking}", flush=True)

        for _ in range(warmups):
            one_run(features, n_steps, tracking, interval)

        measured = []
        for rep in range(repeats):
            result = one_run(features, n_steps, tracking, interval)
            result.update({"sample_count": n, "feature_dimension": d, "repeat": rep + 1})
            measured.append(result)
            raw_rows.append(result)
            print(
                f"  repeat {rep+1}/{repeats}: total={result['total_s']:.4f}s "
                f"similarity={result['similarity_s']:.4f}s pcscs={result['pcscs_s']:.4f}s",
                flush=True,
            )

        conditions[str(n)] = {
            "sample_count": n,
            "feature_dimension": d,
            "feature_sha256": feature_sha,
            "total_s": stats([x["total_s"] for x in measured]),
            "similarity_s": stats([x["similarity_s"] for x in measured]),
            "pcscs_s": stats([x["pcscs_s"] for x in measured]),
            "peak_process_rss_bytes": stats([
                float(x["telemetry_summary"]["peak_process_rss_bytes"])
                for x in measured
                if x["telemetry_summary"].get("peak_process_rss_bytes") is not None
            ]),
        }

    import pcscs

    result_doc = {
        "schema_version": 1,
        "benchmark_id": cfg["benchmark_id"],
        "config_sha256": sha256_file(config_path),
        "configuration": cfg,
        "git": git_info(),
        "environment": runtime_environment(),
        "pcscs_version": getattr(pcscs, "__version__", "unknown"),
        "conditions": conditions,
        "raw_runs": raw_rows,
    }
    dump_json(output_dir / "benchmark_results.json", result_doc)

    csv_path = output_dir / "benchmark_runs.csv"
    fields = [
        "sample_count", "feature_dimension", "repeat", "total_s", "similarity_s", "pcscs_s",
        "convergence_threshold", "critical_threshold", "critical_rate", "fit_successful",
        "peak_process_rss_bytes", "mean_process_cpu_percent", "peak_process_cpu_percent",
        "mean_gpu_utilization_percent", "peak_gpu_utilization_percent", "peak_gpu_memory_used_bytes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in raw_rows:
            telem = row["telemetry_summary"]
            flat = {k: row.get(k) for k in fields}
            for k in fields:
                if k in telem:
                    flat[k] = telem[k]
            writer.writerow(flat)

    return result_doc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--sample-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Optional override for smoke tests or alternate runs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args.config, args.output_dir, args.sample_sizes)
    print(json.dumps({"benchmark_id": result["benchmark_id"], "conditions": list(result["conditions"])}, indent=2))


if __name__ == "__main__":
    main()
