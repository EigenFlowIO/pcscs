#!/usr/bin/env python3
"""Definitive repository-native PCSCS empirical analysis on the frozen probe set.

This is the authoritative scientific runner for the human-curated static sample.
It never queries GBIF and never replaces sample images.  It exercises the full
PCSCS analysis on all configured VGG16 convolutional layers, preserves the rich
tracking/hierarchy outputs needed for human representation analysis, runs the
package spectral extension at a bounded set of thresholds, captures performance
telemetry, and optionally runs the controlled scaling benchmark.

The expected execution pattern is: clone a pinned repository commit in Colab,
install that checkout, verify ``static_dataset``, then execute this script.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATASET = HERE / "static_dataset"
DEFAULT_OUTPUT = Path("pcscs_static_empirical_analysis_bundle.zip")


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
        raise RuntimeError("static dataset has no checksums.sha256")
    errors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        target = dataset_dir / rel
        if not target.exists():
            errors.append(f"missing {rel}")
        elif sha256_file(target) != expected:
            errors.append(f"hash mismatch {rel}")
    if errors:
        raise RuntimeError("static dataset checksum verification failed: " + "; ".join(errors[:20]))


def make_zip(source_dir: Path, output: Path) -> None:
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(source_dir))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_cross_layer_outputs(
    cfg: dict[str, Any],
    manifest_path: Path,
    results_dir: Path,
    extended_dir: Path,
    cross_dir: Path,
) -> None:
    """Generate cross-layer analysis products from the preserved per-layer outputs."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cross_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(manifest_path)
    rows = sorted(manifest["samples"], key=lambda r: r["sample_index"])
    n = len(rows)
    layer_names = list(cfg["model"]["layers"])

    # --- Pairwise similarity evolution across all layers --------------------
    iu = np.triu_indices(n, k=1)
    pair_stack = []
    for layer in layer_names:
        with np.load(results_dir / f"{layer.replace('.', '_')}.npz") as z:
            pair_stack.append(z["similarity_matrix"][iu].astype(np.float32))
    pair_stack = np.stack(pair_stack, axis=0)  # layers x pairs
    pmin = pair_stack.min(axis=0)
    pmax = pair_stack.max(axis=0)
    prange = pmax - pmin
    pdelta = pair_stack[-1] - pair_stack[0]
    pmean = pair_stack.mean(axis=0)
    pstd = pair_stack.std(axis=0)
    min_layer = pair_stack.argmin(axis=0)
    max_layer = pair_stack.argmax(axis=0)

    pair_csv = cross_dir / "all_pairwise_similarity_evolution.csv"
    pair_rows = []
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "sample_i", "sample_j", "sample_id_i", "sample_id_j",
            "family_i", "family_j", "similarity_first_layer", "similarity_last_layer",
            "delta_last_minus_first", "min_similarity", "max_similarity", "range",
            "mean_similarity", "std_similarity", "layer_of_min", "layer_of_max",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k, (i, j) in enumerate(zip(iu[0], iu[1])):
            rec = {
                "sample_i": int(i), "sample_j": int(j),
                "sample_id_i": rows[i].get("sample_id", f"sample_{i:04d}"),
                "sample_id_j": rows[j].get("sample_id", f"sample_{j:04d}"),
                "family_i": rows[i]["family"], "family_j": rows[j]["family"],
                "similarity_first_layer": float(pair_stack[0, k]),
                "similarity_last_layer": float(pair_stack[-1, k]),
                "delta_last_minus_first": float(pdelta[k]),
                "min_similarity": float(pmin[k]), "max_similarity": float(pmax[k]),
                "range": float(prange[k]), "mean_similarity": float(pmean[k]),
                "std_similarity": float(pstd[k]),
                "layer_of_min": layer_names[int(min_layer[k])],
                "layer_of_max": layer_names[int(max_layer[k])],
            }
            w.writerow(rec)
            pair_rows.append(rec)

    # Select informative traces deterministically: largest change range, largest
    # increase, largest decrease, and highest stable similarities.
    selected_indices: list[int] = []
    def add_indices(order, count=15):
        for x in order[:count]:
            x = int(x)
            if x not in selected_indices:
                selected_indices.append(x)

    add_indices(np.argsort(-prange), 20)
    add_indices(np.argsort(-pdelta), 15)
    add_indices(np.argsort(pdelta), 15)
    stable_score = pmean - 2.0 * pstd
    add_indices(np.argsort(-stable_score), 15)

    with (cross_dir / "selected_pairwise_similarity_traces.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["pair_rank", "sample_i", "sample_j", "sample_id_i", "sample_id_j", "family_i", "family_j", "layer", "similarity"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for rank, k in enumerate(selected_indices):
            i, j = int(iu[0][k]), int(iu[1][k])
            for layer_idx, layer in enumerate(layer_names):
                w.writerow({
                    "pair_rank": rank, "sample_i": i, "sample_j": j,
                    "sample_id_i": rows[i].get("sample_id", f"sample_{i:04d}"),
                    "sample_id_j": rows[j].get("sample_id", f"sample_{j:04d}"),
                    "family_i": rows[i]["family"], "family_j": rows[j]["family"],
                    "layer": layer, "similarity": float(pair_stack[layer_idx, k]),
                })

    # Plot a manageable subset of largest-range traces for manuscript triage.
    fig, ax = plt.subplots(figsize=(13, 7))
    for k in selected_indices[:12]:
        i, j = int(iu[0][k]), int(iu[1][k])
        ax.plot(range(len(layer_names)), pair_stack[:, k], marker="o", markersize=2,
                label=f"{i}-{j} {rows[i]['family'][:5]}/{rows[j]['family'][:5]}")
    ax.set_xticks(range(len(layer_names))); ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Selected probe-pair similarity trajectories across VGG16 depth")
    ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(cross_dir / "selected_pairwise_similarity_traces.png", dpi=180); plt.close(fig)

    # --- Component-count trajectory geometry summaries ----------------------
    shape_rows = []
    fig, ax = plt.subplots(figsize=(12, 7))
    for layer in layer_names:
        with np.load(results_dir / f"{layer.replace('.', '_')}.npz") as z:
            thresholds = z["thresholds"].astype(float)
            counts = z["num_components"].astype(float)
            smooth_t = z["smooth_thresholds"].astype(float)
            derivative = z["derivative"].astype(float)
        changed = np.diff(counts) != 0
        plateau_fraction = float(1.0 - changed.mean()) if len(changed) else 1.0
        largest_drop = float(np.min(np.diff(counts))) if len(counts) > 1 else 0.0
        total_drop = max(float(counts[0] - counts[-1]), 1.0)
        max_abs = float(np.nanmax(np.abs(derivative))) if len(derivative) else 0.0
        half_width = 0.0
        if max_abs > 0:
            mask = np.abs(derivative) >= 0.5 * max_abs
            if np.any(mask):
                vals = smooth_t[mask]
                half_width = float(abs(vals.max() - vals.min()))
        # AUC of normalized component count over the actual threshold interval.
        norm_counts = (counts - counts[-1]) / total_drop
        auc = float(abs(np.trapz(norm_counts, thresholds)) / max(abs(thresholds[0]-thresholds[-1]), 1e-12))
        shape_rows.append({
            "layer": layer,
            "threshold_span": float(abs(thresholds[0] - thresholds[-1])),
            "raw_change_steps": int(changed.sum()),
            "plateau_fraction": plateau_fraction,
            "largest_single_step_component_change": largest_drop,
            "half_max_derivative_width": half_width,
            "normalized_component_curve_auc": auc,
        })
        ax.plot(thresholds, counts, linewidth=1.1, label=layer)
    with (cross_dir / "component_trajectory_shape_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(shape_rows[0]); w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(shape_rows)
    ax.set_xlabel("Cosine similarity threshold"); ax.set_ylabel("Connected components")
    ax.set_title("PCSCS component-count trajectories across VGG16 layers")
    ax.grid(alpha=.2); ax.legend(fontsize=6, ncol=3)
    fig.tight_layout(); fig.savefig(cross_dir / "all_layer_component_trajectories.png", dpi=180); plt.close(fig)

    # --- Critical/convergence metrics across depth --------------------------
    metrics = list(csv.DictReader((results_dir / "layer_metrics.csv").open(encoding="utf-8")))
    x = np.arange(len(metrics))
    critical = np.array([float(r["critical_threshold"]) for r in metrics])
    convergence = np.array([float(r["convergence_threshold"]) for r in metrics])
    rate = np.array([float(r["critical_rate"]) for r in metrics])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, critical, marker="o", label="critical threshold")
    ax.plot(x, convergence, marker="o", label="convergence threshold")
    ax.set_xticks(x); ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_ylabel("Cosine similarity threshold"); ax.set_title("PCSCS threshold landmarks across VGG16 depth")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(cross_dir / "threshold_landmarks_across_layers.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, rate, marker="o")
    ax.set_xticks(x); ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_ylabel("Critical rate"); ax.set_title("Maximum component-count rate across VGG16 depth")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(cross_dir / "critical_rate_across_layers.png", dpi=180); plt.close(fig)

    # --- Individual-sample first-merge trajectories across depth ------------
    with (cross_dir / "sample_first_merge_across_layers.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_index", "sample_id", "family"] + layer_names
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        per_layer = {}
        for layer in layer_names:
            per_layer[layer] = _read_json(extended_dir / layer.replace('.', '_') / "sample_trajectories.json")
        for i, row in enumerate(rows):
            out={"sample_index":i,"sample_id":row.get("sample_id",f"sample_{i:04d}"),"family":row["family"]}
            for layer in layer_names:
                entry=per_layer[layer].get(str(i), {})
                out[layer]=entry.get("first_merge_threshold")
            w.writerow(out)

    _write_json(cross_dir / "cross_layer_output_index.json", {
        "layers": layer_names,
        "n_samples": n,
        "n_pairs": int(len(iu[0])),
        "selected_pair_count": len(selected_indices),
        "files": sorted(p.name for p in cross_dir.iterdir() if p.is_file()),
    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--work-root", type=Path, default=Path("static_empirical_analysis_work"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--telemetry-interval", type=float, default=1.0)
    p.add_argument("--spectral-steps", type=int, default=24,
                   help="Number of thresholds per layer for the supporting dynamic spectral analysis; 0 disables it.")
    p.add_argument("--skip-benchmark", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    work_root = args.work_root.resolve()
    results_dir = work_root / "results"
    scratch_dir = work_root / "scratch"
    benchmark_dir = work_root / "benchmark_results"
    extended_dir = work_root / "extended_analysis"
    cross_dir = work_root / "cross_layer_analysis"
    bundle_dir = work_root / "bundle"
    data_dir = work_root / "data"

    runner = load_runner()
    cfg_path = HERE / "config.json"
    manifest_src = dataset_dir / "sample-manifest.json"
    images_dir = dataset_dir / "images"

    status: dict[str, Any] = {
        "schema_version": 2,
        "validation_name": "PCSCS definitive static empirical analysis",
        "started_utc": runner.utc_now(),
        "success": False,
        "dataset_mode": "committed_human_curated_static_images",
        "network_sample_acquisition": False,
        "dataset_dir": str(dataset_dir),
        "spectral_steps": int(args.spectral_steps),
        "runtime": {"python": sys.version, "platform": __import__("platform").platform()},
        "stages": [],
    }

    monitor = None
    telemetry = None
    try:
        if work_root.exists():
            shutil.rmtree(work_root)
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_src, data_dir / "sample-manifest.json")
        for name in ["dataset-metadata.json", "curation-manifest.json", "curation-config.json", "human_rejections.csv", "automatic_skips.csv", "IMAGE_LICENSES.csv"]:
            src = dataset_dir / name
            if src.exists(): shutil.copy2(src, data_dir / name)

        verify_checksums_file(dataset_dir)
        cfg = runner.load_json(cfg_path)
        runner.validate_sample(cfg, cfg_path, manifest_src, images_dir)
        status["stages"].append({"stage": "verify_static_dataset", "ok": True})

        from pcscs.performance import PerformanceMonitor
        monitor = PerformanceMonitor(interval_s=args.telemetry_interval).start()

        with monitor.phase("scientific_validation", n_samples=cfg["target_total"]):
            runner.run_pcscs_validation(
                cfg, cfg_path, manifest_src, images_dir, results_dir, scratch_dir,
                status, monitor, extended_dir=extended_dir, spectral_steps=args.spectral_steps,
            )
        status["stages"].append({"stage": "run_full_pcscs_validation", "ok": True})

        with monitor.phase("result_validation"):
            runner.validate_results(cfg, cfg_path, manifest_src, results_dir)
        status["stages"].append({"stage": "validate_results", "ok": True})

        with monitor.phase("cross_layer_analysis"):
            build_cross_layer_outputs(cfg, manifest_src, results_dir, extended_dir, cross_dir)
        status["stages"].append({"stage": "cross_layer_analysis", "ok": True})

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
        status["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        if monitor is not None:
            monitor.stop(); telemetry = monitor.to_dict()

        if bundle_dir.exists(): shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        runner.collect_bundle(REPO_ROOT, cfg_path, data_dir, results_dir, benchmark_dir, bundle_dir, status, telemetry)

        # Full analytical outputs are intentionally first-class bundle contents.
        if extended_dir.exists(): shutil.copytree(extended_dir, bundle_dir / "extended_analysis")
        if cross_dir.exists(): shutil.copytree(cross_dir, bundle_dir / "cross_layer_analysis")
        for name in ["dataset-metadata.json", "curation-manifest.json", "curation-config.json", "human_rejections.csv", "automatic_skips.csv", "IMAGE_LICENSES.csv"]:
            src = data_dir / name
            if src.exists(): shutil.copy2(src, bundle_dir / name)
        shutil.copy2(dataset_dir / "checksums.sha256", bundle_dir / "static-dataset-checksums.sha256")
        (bundle_dir / "STATIC_DATASET_MODE.txt").write_text(
            "This definitive run used only the human-curated sample images committed in the pinned repository checkout. No GBIF sample query, specimen image retrieval, or sample replacement occurred during analysis.\n",
            encoding="utf-8",
        )
        (bundle_dir / "INGEST_ME.txt").write_text(
            "Upload this bundle to ChatGPT unchanged. It contains the complete static PCSCS scientific outputs, full tracking/merge histories, filtration-faithful hierarchy data and dendrograms, critical-component image grids, cross-layer pairwise and sample trajectories, supporting spectral outputs, performance telemetry, controlled scaling results (unless explicitly skipped), environment metadata, source hashes, and checksums.\n",
            encoding="utf-8",
        )
        runner.write_checksums(bundle_dir)
        output = args.output.resolve()
        make_zip(bundle_dir, output)
        print(f"FINAL STATIC EMPIRICAL ANALYSIS BUNDLE: {output}")
        print(f"SHA-256: {sha256_file(output)}")

    if not status["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
