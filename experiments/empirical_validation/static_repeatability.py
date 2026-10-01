#!/usr/bin/env python3
"""Repeat the PCSCS scientific validation from the committed static image set.

Unlike ``pcscs_colab_validation.py``, this runner never queries GBIF and never
retrieves sample images. The exact sample is the committed static dataset:
``static_dataset/sample-manifest.json`` plus ``static_dataset/images``.

It reuses the scientific-analysis functions from the authoritative validation
runner so there is only one implementation of feature extraction and PCSCS.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATASET = HERE / "static_dataset"
DEFAULT_OUTPUT = Path("pcscs_static_repeatability_bundle.zip")


def load_runner():
    path = HERE / "pcscs_colab_validation.py"
    spec = importlib.util.spec_from_file_location("pcscs_validation_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums_file(dataset_dir: Path) -> None:
    path = dataset_dir / "checksums.sha256"
    if not path.exists():
        raise RuntimeError("static dataset has no checksums.sha256; run cache_static_dataset.py first")
    errors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        target = dataset_dir / rel
        if not target.exists():
            errors.append(f"missing {rel}")
        else:
            actual = sha256_file(target)
            if actual != expected:
                errors.append(f"hash mismatch {rel}")
    if errors:
        raise RuntimeError("static dataset checksum verification failed: " + "; ".join(errors[:10]))


def make_zip(source_dir: Path, output: Path) -> None:
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(source_dir))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--work-root", type=Path, default=Path("static_repeatability_work"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--telemetry-interval", type=float, default=1.0)
    p.add_argument("--skip-benchmark", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    work_root = args.work_root.resolve()
    results_dir = work_root / "results"
    scratch_dir = work_root / "scratch"
    benchmark_dir = work_root / "benchmark_results"
    bundle_dir = work_root / "bundle"
    data_dir = work_root / "data"

    runner = load_runner()
    cfg_path = HERE / "config.json"
    manifest_src = dataset_dir / "sample-manifest.json"
    images_dir = dataset_dir / "images"

    status: dict[str, Any] = {
        "schema_version": 1,
        "validation_name": "PCSCS static repeatability validation",
        "started_utc": runner.utc_now(),
        "success": False,
        "dataset_mode": "committed_static_images",
        "network_sample_acquisition": False,
        "dataset_dir": str(dataset_dir),
        "stages": [],
    }

    monitor = None
    telemetry = None
    try:
        if work_root.exists():
            shutil.rmtree(work_root)
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_src, data_dir / "sample-manifest.json")
        metadata_src = dataset_dir / "dataset-metadata.json"
        if metadata_src.exists():
            shutil.copy2(metadata_src, data_dir / "dataset-metadata.json")

        verify_checksums_file(dataset_dir)
        runner.validate_sample(runner.load_json(cfg_path), cfg_path, manifest_src, images_dir)
        status["stages"].append({"stage": "verify_static_dataset", "ok": True})

        from pcscs.performance import PerformanceMonitor
        monitor = PerformanceMonitor(interval_s=args.telemetry_interval).start()
        cfg = runner.load_json(cfg_path)

        with monitor.phase("scientific_validation"):
            runner.run_pcscs_validation(
                cfg, cfg_path, manifest_src, images_dir, results_dir, scratch_dir, status, monitor
            )
        status["stages"].append({"stage": "run_pcscs_validation", "ok": True})

        with monitor.phase("result_validation"):
            runner.validate_results(cfg, cfg_path, manifest_src, results_dir)
        status["stages"].append({"stage": "validate_results", "ok": True})

        if not args.skip_benchmark:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "benchmarks" / "benchmark_pcscs.py"),
                "--config", str(REPO_ROOT / "benchmarks" / "config.json"),
                "--output-dir", str(benchmark_dir),
            ]
            with monitor.phase("controlled_scaling_benchmark"):
                runner.run(cmd, cwd=REPO_ROOT)
            status["stages"].append({"stage": "controlled_scaling_benchmark", "ok": True})

        status["success"] = True
    except Exception as exc:
        status["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()
    finally:
        if monitor is not None:
            monitor.stop()
            telemetry = monitor.to_dict()

        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Reuse standard scientific bundle collection, but identify this as a static run.
        runner.collect_bundle(
            REPO_ROOT,
            cfg_path,
            data_dir,
            results_dir,
            benchmark_dir,
            bundle_dir,
            status,
            telemetry,
        )
        if metadata_src.exists():
            shutil.copy2(metadata_src, bundle_dir / "dataset-metadata.json")
        shutil.copy2(dataset_dir / "checksums.sha256", bundle_dir / "static-dataset-checksums.sha256")
        (bundle_dir / "STATIC_DATASET_MODE.txt").write_text(
            "This run used only the committed static sample images. No GBIF sample query or image retrieval was performed.\n",
            encoding="utf-8",
        )
        runner.write_checksums(bundle_dir)
        make_zip(bundle_dir, args.output.resolve())
        print(f"FINAL STATIC REPEATABILITY BUNDLE: {args.output.resolve()}")
        print(f"SHA-256: {sha256_file(args.output.resolve())}")

    if not status["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
