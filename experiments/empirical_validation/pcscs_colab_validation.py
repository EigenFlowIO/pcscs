#!/usr/bin/env python3
"""
PCSCS definitive empirical validation runner for Google Colab.

This is a self-contained validation driver. It:
1. clones the public PCSCS GitHub repository,
2. checks out the requested Git ref,
3. installs the package and research dependencies,
4. builds and freezes the configured GBIF specimen sample,
5. validates the exact sample and hashes,
6. extracts the configured VGG16 layer representations,
7. computes cosine-similarity matrices,
8. runs the repository PCSCS implementation,
9. validates the result lineage, and
10. writes one compact ZIP bundle designed for direct ingestion by ChatGPT.

The raw specimen images and high-dimensional activation tensors are intentionally
not included in the final ZIP. The frozen manifest contains the GBIF identifiers,
media URLs, dimensions, and SHA-256 hashes needed to reconstruct the exact sample.

Recommended Colab invocation:
    !python /content/pcscs/experiments/empirical_validation/pcscs_colab_validation.py

Optional:
    !python /content/pcscs/experiments/empirical_validation/pcscs_colab_validation.py --ref <git-ref>

The final artifact is:
    /content/pcscs_validation_bundle.zip
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any

REPO_URL_DEFAULT = "https://github.com/EigenFlowIO/pcscs.git"
WORK_ROOT_DEFAULT = Path("/content/pcscs_validation_work")
BUNDLE_DEFAULT = Path("/content/pcscs_validation_bundle.zip")
USER_AGENT = "PCSCS-validation/1.0"
SEED = 0


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture else ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def suffix_for(content_type: str | None, url: str) -> str:
    ct = (content_type or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "tiff" in ct:
        return ".tif"
    low = url.lower().split("?")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"):
        if low.endswith(ext):
            return ext
    return ".jpg"


def first_still_image(record: dict[str, Any]) -> str | None:
    for media in record.get("media", []):
        if media.get("type") == "StillImage" and media.get("identifier"):
            return media["identifier"]
    return None


def bootstrap_repo(repo_url: str, ref: str, work_root: Path, status: dict[str, Any]) -> Path:
    repo_dir = work_root / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    run(["git", "clone", repo_url, str(repo_dir)])
    run(["git", "checkout", ref], cwd=repo_dir)
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture=True)
    status["repository"] = {
        "url": repo_url,
        "requested_ref": ref,
        "resolved_commit": commit,
    }

    # Install the exact repository checkout into the current interpreter's
    # existing site-packages path. A non-editable install is intentional here:
    # editable installs rely on .pth processing at interpreter startup, which
    # does not occur when pip is invoked from this already-running validation
    # process (as in Colab).
    run([sys.executable, "-m", "pip", "install", "-q", f"{repo_dir}[research,benchmark]"])

    # Fail immediately if the package cannot be imported by this same process.
    # This guards the exact bootstrap failure mode that can otherwise appear
    # only after a successful pip command in a long-running Colab interpreter.
    import importlib
    importlib.invalidate_caches()
    import pcscs
    status["repository"]["installed_pcscs_path"] = str(Path(pcscs.__file__).resolve())
    return repo_dir


def build_candidate_pool(cfg: dict[str, Any], config_path: Path, data_dir: Path) -> Path:
    import requests

    gbif_match = "https://api.gbif.org/v1/species/match"
    gbif_search = "https://api.gbif.org/v1/occurrence/search"
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    config_sha = sha256_file(config_path)
    out: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "config_sha256": config_sha,
        "query_semantics": (
            "Live GBIF provenance snapshot; the frozen validation sample is defined "
            "by sample-manifest.json, not by re-querying GBIF."
        ),
        "families": [],
    }

    limit = int(cfg["candidate_pool_per_family"])
    for fam in cfg["families"]:
        family = fam["family"]
        m = session.get(gbif_match, params={"name": family}, timeout=30)
        m.raise_for_status()
        usage_key = m.json().get("usageKey")
        if not usage_key:
            raise RuntimeError(f"No GBIF usageKey for {family}")

        params = {
            "taxonKey": usage_key,
            "mediaType": cfg["gbif"]["mediaType"],
            "basisOfRecord": cfg["gbif"]["basisOfRecord"],
            "limit": min(limit, 300),
        }
        r = session.get(gbif_search, params=params, timeout=60)
        r.raise_for_status()

        candidates = []
        for rec in r.json().get("results", []):
            url = first_still_image(rec)
            key = rec.get("key")
            if key is None or not url:
                continue
            candidates.append(
                {
                    "occurrence_key": int(key),
                    "media_url": url,
                    "scientific_name": rec.get("scientificName"),
                    "species": rec.get("species"),
                    "institution_code": rec.get("institutionCode"),
                    "collection_code": rec.get("collectionCode"),
                    "catalog_number": rec.get("catalogNumber"),
                }
            )

        candidates.sort(key=lambda x: (x["occurrence_key"], x["media_url"]))
        out["families"].append({**fam, "gbif_usage_key": usage_key, "candidates": candidates})
        print(f"{family}: {len(candidates)} candidates", flush=True)

    candidate_pool_path = data_dir / "candidate-pool.json"
    dump_json(candidate_pool_path, out)
    return candidate_pool_path


def freeze_sample(
    cfg: dict[str, Any],
    config_path: Path,
    candidate_pool_path: Path,
    data_dir: Path,
) -> tuple[Path, Path]:
    import requests
    from PIL import Image

    pool = load_json(candidate_pool_path)
    config_sha = sha256_file(config_path)
    if pool.get("config_sha256") != config_sha:
        raise RuntimeError("Candidate pool was built with a different config.json")

    target = int(cfg["target_per_family"])
    min_w = int(cfg["minimum_width"])
    min_h = int(cfg["minimum_height"])
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    selected: list[dict[str, Any]] = []

    for fam in pool["families"]:
        family = fam["family"]
        accepted = 0

        for cand in sorted(
            fam["candidates"],
            key=lambda x: (x["occurrence_key"], x["media_url"]),
        ):
            if accepted >= target:
                break
            try:
                r = session.get(cand["media_url"], timeout=45)
                r.raise_for_status()
                raw = r.content

                im = Image.open(io.BytesIO(raw))
                im.verify()
                im = Image.open(io.BytesIO(raw))
                width, height = im.size
                if width < min_w or height < min_h:
                    continue

                sha = sha256_bytes(raw)
                ext = suffix_for(r.headers.get("Content-Type"), cand["media_url"])
                filename = f"{family}__{cand['occurrence_key']}__{sha[:12]}{ext}"
                (image_dir / filename).write_bytes(raw)

                selected.append(
                    {
                        "sample_index": len(selected),
                        "family": family,
                        "imagenet_class": fam["imagenet_class"],
                        "imagenet_index": fam["imagenet_index"],
                        **cand,
                        "filename": filename,
                        "sha256": sha,
                        "width": width,
                        "height": height,
                    }
                )
                accepted += 1
                print(
                    f"{family}: accepted {accepted}/{target} occurrence "
                    f"{cand['occurrence_key']}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"{family}: rejected occurrence {cand.get('occurrence_key')}: {exc}",
                    flush=True,
                )

        if accepted != target:
            raise RuntimeError(
                f"{family}: only {accepted}/{target} valid specimens; "
                "the fixed design was not silently changed"
            )

    if len(selected) != int(cfg["target_total"]):
        raise RuntimeError(
            f"Expected {cfg['target_total']} samples, got {len(selected)}"
        )

    manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "frozen_utc": utc_now(),
        "config_sha256": config_sha,
        "candidate_pool_generated_utc": pool.get("generated_utc"),
        "selection_rule": (
            "For each family, sort archived candidates by "
            "(occurrence_key, media_url), then accept the first "
            "target_per_family images that download, decode, and meet "
            "the fixed minimum dimensions."
        ),
        "target_total": cfg["target_total"],
        "target_per_family": target,
        "samples": selected,
    }
    manifest_path = data_dir / "sample-manifest.json"
    dump_json(manifest_path, manifest)
    return manifest_path, image_dir


def validate_sample(
    cfg: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    image_dir: Path,
) -> None:
    from collections import Counter
    from PIL import Image

    manifest = load_json(manifest_path)
    errors: list[str] = []

    if manifest.get("config_sha256") != sha256_file(config_path):
        errors.append("config hash mismatch")

    samples = manifest.get("samples", [])
    if len(samples) != int(cfg["target_total"]):
        errors.append(f"sample count {len(samples)} != {cfg['target_total']}")

    ids = [x["occurrence_key"] for x in samples]
    if len(ids) != len(set(ids)):
        errors.append("duplicate GBIF occurrence keys")

    counts = Counter(x["family"] for x in samples)
    expected_fams = {f["family"] for f in cfg["families"]}
    if set(counts) != expected_fams:
        errors.append("family set mismatch")

    for fam in expected_fams:
        if counts[fam] != int(cfg["target_per_family"]):
            errors.append(f"{fam}: {counts[fam]} samples")

    for row in samples:
        path = image_dir / row["filename"]
        if not path.exists():
            errors.append(f"missing {row['filename']}")
            continue
        if sha256_file(path) != row["sha256"]:
            errors.append(f"hash mismatch {row['filename']}")
        try:
            with Image.open(path) as im:
                if im.width != row["width"] or im.height != row["height"]:
                    errors.append(f"dimension mismatch {row['filename']}")
                if im.width < cfg["minimum_width"] or im.height < cfg["minimum_height"]:
                    errors.append(f"below minimum size {row['filename']}")
        except Exception as exc:
            errors.append(f"decode failure {row['filename']}: {exc}")

    if errors:
        raise RuntimeError("Sample validation FAILED:\n- " + "\n- ".join(errors))

    print(
        f"Sample validation PASSED: {len(samples)} specimens; "
        f"{len(counts)} families; exact hashes verified",
        flush=True,
    )


def configure_determinism(torch_module: Any) -> None:
    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(SEED)
    torch_module.manual_seed(SEED)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(SEED)
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def get_layer(model: Any, name: str) -> Any:
    layer = model
    for part in name.split("."):
        layer = getattr(layer, part)
    return layer


def extract_all_layer_features_to_disk(
    cfg: dict[str, Any],
    manifest_path: Path,
    image_dir: Path,
    scratch_dir: Path,
    status: dict[str, Any],
    monitor: Any,
) -> dict[str, dict[str, Any]]:
    """Extract VGG16 representations once, preserving exact flattened activations."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.models import VGG16_Weights, vgg16

    configure_determinism(torch)

    weights_name = cfg["model"]["torchvision_weights"]
    if weights_name != "IMAGENET1K_V1":
        raise RuntimeError(f"Unsupported configured VGG16 weights: {weights_name}")

    weights = VGG16_Weights.IMAGENET1K_V1
    with monitor.phase("model_initialization", architecture="vgg16", weights=weights_name):
        model = vgg16(weights=weights)
        preprocess = weights.transforms()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

    status["runtime"]["device"] = device
    if torch.cuda.is_available():
        status["runtime"]["cuda_device_name"] = torch.cuda.get_device_name(0)

    rows = sorted(load_json(manifest_path)["samples"], key=lambda x: x["sample_index"])
    layer_names = list(cfg["model"]["layers"])
    activations: dict[str, Any] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            activations[name] = output.detach()
        return hook

    for name in layer_names:
        hooks.append(get_layer(model, name).register_forward_hook(make_hook(name)))

    feature_meta: dict[str, dict[str, Any]] = {}
    memmaps: dict[str, Any] = {}
    materialization_s = {name: 0.0 for name in layer_names}
    forward_total_s = 0.0
    preprocessing_total_s = 0.0

    try:
        for sample_idx, row in enumerate(rows):
            path = image_dir / row["filename"]
            t0 = __import__("time").perf_counter()
            with Image.open(path) as im:
                image = im.convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)
            preprocessing_total_s += __import__("time").perf_counter() - t0

            activations.clear()
            t0 = __import__("time").perf_counter()
            with torch.no_grad():
                _ = model(tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            forward_total_s += __import__("time").perf_counter() - t0

            for layer_name in layer_names:
                if layer_name not in activations:
                    raise RuntimeError(f"No activation captured for {layer_name}")
                t0 = __import__("time").perf_counter()
                raw_shape = list(activations[layer_name].shape)
                flat = (
                    activations[layer_name]
                    .flatten(start_dim=1)
                    .squeeze(0)
                    .to("cpu", dtype=torch.float32)
                    .numpy()
                )
                if layer_name not in memmaps:
                    layer_dir = scratch_dir / "features"
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    mmap_path = layer_dir / f"{layer_name.replace('.', '_')}.f32"
                    mm = np.memmap(
                        mmap_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(rows), flat.size),
                    )
                    memmaps[layer_name] = mm
                    feature_meta[layer_name] = {
                        "path": str(mmap_path),
                        "shape": [len(rows), int(flat.size)],
                        "activation_shape": raw_shape,
                        "dtype": "float32",
                    }
                memmaps[layer_name][sample_idx, :] = flat
                materialization_s[layer_name] += __import__("time").perf_counter() - t0

            if sample_idx % 10 == 0 or sample_idx + 1 == len(rows):
                print(f"VGG16 extraction: {sample_idx + 1}/{len(rows)} samples", flush=True)

            activations.clear()
            del tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for mm in memmaps.values():
            mm.flush()
    finally:
        for h in hooks:
            h.remove()

    monitor.phases.append({
        "name": "image_preprocessing",
        "duration_s": float(preprocessing_total_s),
        "ok": True,
        "metadata": {"n_samples": len(rows)},
    })
    monitor.phases.append({
        "name": "vgg16_forward_pass",
        "duration_s": float(forward_total_s),
        "ok": True,
        "metadata": {"n_samples": len(rows), "device": device},
    })
    for layer_name in layer_names:
        feature_meta[layer_name]["materialization_s"] = float(materialization_s[layer_name])
        monitor.add_layer_metric(
            layer_name,
            activation_shape=feature_meta[layer_name]["activation_shape"],
            flattened_dimension=int(feature_meta[layer_name]["shape"][1]),
            materialization_s=float(materialization_s[layer_name]),
        )

    memmaps.clear()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return feature_meta

def normalize_memmap_in_place(path: Path, shape: tuple[int, int], block_rows: int = 4) -> None:
    import numpy as np

    mm = np.memmap(path, mode="r+", dtype=np.float32, shape=shape)
    n = shape[0]
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        block = np.asarray(mm[start:stop], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mm[start:stop] = block / norms
    mm.flush()
    del mm


def similarity_from_normalized_memmap(
    path: Path,
    shape: tuple[int, int],
    device: str,
    block_rows: int = 8,
):
    """
    Compute the full cosine-similarity matrix from normalized flattened
    representations in bounded-memory blocks. GPU is used for block matrix
    products when available; CPU remains supported.
    """
    import numpy as np
    import torch

    mm = np.memmap(path, mode="r", dtype=np.float32, shape=shape)
    n = shape[0]
    sim = np.empty((n, n), dtype=np.float32)

    for i in range(0, n, block_rows):
        i2 = min(i + block_rows, n)
        a_np = np.asarray(mm[i:i2], dtype=np.float32)

        if device == "cuda":
            a = torch.from_numpy(a_np.copy()).to(device)
        else:
            a = None

        for j in range(i, n, block_rows):
            j2 = min(j + block_rows, n)
            b_np = np.asarray(mm[j:j2], dtype=np.float32)

            if device == "cuda":
                b = torch.from_numpy(b_np.copy()).to(device)
                block = (a @ b.T).cpu().numpy()
                del b
            else:
                block = np.dot(a_np, b_np.T)

            sim[i:i2, j:j2] = block
            if j != i:
                sim[j:j2, i:i2] = block.T

        if device == "cuda":
            del a
            torch.cuda.empty_cache()

    del mm
    # Exact self-similarity is useful for validation and removes tiny numerical
    # diagonal drift from blocked dot products.
    np.fill_diagonal(sim, 1.0)
    return sim


def jsonable_label(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": row["family"],
        "imagenet_class": row["imagenet_class"],
        "imagenet_index": row["imagenet_index"],
        "occurrence_key": row["occurrence_key"],
    }


def run_pcscs_validation(
    cfg: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    image_dir: Path,
    results_dir: Path,
    scratch_dir: Path,
    status: dict[str, Any],
    monitor: Any,
) -> None:
    import numpy as np
    import torch
    import torchvision
    import pcscs
    from pcscs import PCSCS
    from pcscs.performance import reset_torch_cuda_peak_memory, torch_cuda_memory

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    rows = sorted(manifest["samples"], key=lambda x: x["sample_index"])
    labels = [jsonable_label(r) for r in rows]

    with monitor.phase("feature_extraction_and_materialization", n_samples=len(rows)):
        feature_meta = extract_all_layer_features_to_disk(
            cfg, manifest_path, image_dir, scratch_dir, status, monitor
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "n_samples": len(rows),
        "seed": SEED,
        "device": status["runtime"].get("device"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pcscs_version": getattr(pcscs, "__version__", "unknown"),
        "torchvision_weights": cfg["model"]["torchvision_weights"],
        "layers": {},
    }

    metrics_rows: list[dict[str, Any]] = []
    layer_names = list(cfg["model"]["layers"])

    for layer_index, layer_name in enumerate(layer_names, start=1):
        meta = feature_meta[layer_name]
        mmap_path = Path(meta["path"])
        shape = tuple(meta["shape"])
        reset_torch_cuda_peak_memory()

        with monitor.layer_phase(
            layer_name,
            "normalize_features",
            layer_index=layer_index,
            activation_shape=meta["activation_shape"],
            flattened_dimension=int(shape[1]),
        ):
            normalize_memmap_in_place(mmap_path, shape)

        with monitor.layer_phase(
            layer_name,
            "cosine_similarity",
            layer_index=layer_index,
            flattened_dimension=int(shape[1]),
        ) as sim_record:
            sim = similarity_from_normalized_memmap(
                mmap_path,
                shape,
                status["runtime"]["device"],
            )
            sim_record.update(torch_cuda_memory())

        analyzer = PCSCS(enable_tracking=bool(cfg["pcscs"]["enable_tracking"]))
        with monitor.layer_phase(
            layer_name,
            "pcscs_analysis",
            layer_index=layer_index,
            n_steps=int(cfg["pcscs"]["n_steps"]),
            tracking=bool(cfg["pcscs"]["enable_tracking"]),
        ) as pcscs_record:
            result = analyzer.analyze_layer(
                sim,
                layer_name=layer_name,
                sample_labels=labels,
                n_steps=int(cfg["pcscs"]["n_steps"]),
            )
            pcscs_record.update(torch_cuda_memory())

        layer_key = layer_name.replace(".", "_")
        with monitor.layer_phase(layer_name, "result_serialization", layer_index=layer_index):
            np.savez_compressed(
                results_dir / f"{layer_key}.npz",
                similarity_matrix=sim,
                thresholds=result.thresholds,
                num_components=result.num_classes,
                smooth_thresholds=result.smooth_thresholds,
                smooth_components=result.smooth_classes,
                derivative=result.derivative,
                sigmoid_params=(
                    np.asarray(result.sigmoid_params)
                    if result.sigmoid_params is not None
                    else np.asarray([], dtype=float)
                ),
            )

        layer_records = [x for x in monitor.layers if x.get("layer") == layer_name]
        phase_times = {
            x["phase"]: x.get("duration_s")
            for x in layer_records
            if x.get("duration_s") is not None
        }
        row = {
            "layer": layer_name,
            "layer_index": layer_index,
            "activation_shape": json.dumps(meta["activation_shape"]),
            "feature_dimension": int(shape[1]),
            "materialization_s": float(meta.get("materialization_s", 0.0)),
            "normalization_s": float(phase_times.get("normalize_features", 0.0)),
            "similarity_s": float(phase_times.get("cosine_similarity", 0.0)),
            "pcscs_s": float(phase_times.get("pcscs_analysis", 0.0)),
            "serialization_s": float(phase_times.get("result_serialization", 0.0)),
            "convergence_threshold": float(result.convergence_threshold),
            "critical_threshold": float(result.critical_threshold),
            "critical_rate": float(result.critical_rate),
            "fit_successful": bool(result.fit_successful),
        }
        row["layer_total_s"] = float(
            row["materialization_s"] + row["normalization_s"] + row["similarity_s"] + row["pcscs_s"] + row["serialization_s"]
        )
        summary["layers"][layer_name] = {k: v for k, v in row.items() if k != "layer"}
        metrics_rows.append(row)

        try:
            mmap_path.unlink()
        except OSError:
            pass
        del sim, analyzer, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dump_json(results_dir / "summary.json", summary)
    fieldnames = list(metrics_rows[0].keys()) if metrics_rows else []
    with (results_dir / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

def validate_results(
    cfg: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    results_dir: Path,
) -> None:
    import math
    import numpy as np

    summary = load_json(results_dir / "summary.json")
    errors: list[str] = []

    if summary.get("config_sha256") != sha256_file(config_path):
        errors.append("config hash mismatch")
    if summary.get("sample_manifest_sha256") != sha256_file(manifest_path):
        errors.append("sample manifest hash mismatch")
    if summary.get("n_samples") != cfg["target_total"]:
        errors.append("sample count mismatch")

    expected = set(cfg["model"]["layers"])
    got = set(summary.get("layers", {}))
    if got != expected:
        errors.append(
            f"layer mismatch expected={sorted(expected)} got={sorted(got)}"
        )

    for layer, row in summary.get("layers", {}).items():
        for key in (
            "convergence_threshold",
            "critical_threshold",
            "critical_rate",
        ):
            v = row.get(key)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                errors.append(f"{layer}: invalid {key}")

        npz_path = results_dir / f"{layer.replace('.', '_')}.npz"
        if not npz_path.exists():
            errors.append(f"{layer}: missing {npz_path.name}")
            continue

        with np.load(npz_path) as z:
            if "similarity_matrix" not in z.files:
                errors.append(f"{layer}: missing similarity_matrix")
            else:
                sim = z["similarity_matrix"]
                if sim.shape != (cfg["target_total"], cfg["target_total"]):
                    errors.append(f"{layer}: bad similarity matrix shape {sim.shape}")
                if not np.all(np.isfinite(sim)):
                    errors.append(f"{layer}: non-finite similarity values")
                if not np.allclose(sim, sim.T, atol=1e-5):
                    errors.append(f"{layer}: similarity matrix not symmetric")

    if errors:
        raise RuntimeError("Result validation FAILED:\n- " + "\n- ".join(errors))

    print(
        f"Result validation PASSED: {len(got)} layers; "
        "manifest/config lineage verified",
        flush=True,
    )


def write_manifest_csv(manifest_path: Path, out_path: Path) -> None:
    rows = load_json(manifest_path)["samples"]
    fieldnames = [
        "sample_index",
        "family",
        "imagenet_class",
        "imagenet_index",
        "occurrence_key",
        "scientific_name",
        "species",
        "institution_code",
        "collection_code",
        "catalog_number",
        "media_url",
        "filename",
        "sha256",
        "width",
        "height",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_source_hashes(repo_dir: Path, out_path: Path) -> None:
    source_root = repo_dir / "src" / "pcscs"
    hashes = {}
    for path in sorted(source_root.glob("*.py")):
        hashes[str(path.relative_to(repo_dir))] = sha256_file(path)
    dump_json(out_path, hashes)


def write_checksums(bundle_dir: Path) -> None:
    files = [
        p for p in bundle_dir.rglob("*")
        if p.is_file() and p.name != "checksums.sha256"
    ]
    lines = []
    for p in sorted(files):
        rel = p.relative_to(bundle_dir)
        lines.append(f"{sha256_file(p)}  {rel.as_posix()}")
    (bundle_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def collect_bundle(
    repo_dir: Path,
    config_path: Path,
    data_dir: Path,
    results_dir: Path,
    benchmark_dir: Path,
    bundle_dir: Path,
    status: dict[str, Any],
    telemetry: dict[str, Any] | None = None,
) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for src, dst_name in [
        (config_path, "config.json"),
        (data_dir / "candidate-pool.json", "candidate-pool.json"),
        (data_dir / "sample-manifest.json", "sample-manifest.json"),
        (results_dir / "summary.json", "summary.json"),
        (results_dir / "layer_metrics.csv", "layer_metrics.csv"),
    ]:
        if src.exists():
            shutil.copy2(src, bundle_dir / dst_name)

    if (data_dir / "sample-manifest.json").exists():
        write_manifest_csv(data_dir / "sample-manifest.json", bundle_dir / "sample_manifest.csv")

    layer_dir = bundle_dir / "layers"
    layer_dir.mkdir(exist_ok=True)
    if results_dir.exists():
        for p in sorted(results_dir.glob("*.npz")):
            shutil.copy2(p, layer_dir / p.name)

    if telemetry is not None:
        dump_json(bundle_dir / "performance_telemetry.json", telemetry)
        # Flat phase/layer CSVs make the bundle easy to ingest without parsing nested JSON.
        with (bundle_dir / "performance_phases.csv").open("w", newline="", encoding="utf-8") as f:
            rows = telemetry.get("phases", [])
            fields = sorted({k for row in rows for k in row if k != "metadata"})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fields})
        with (bundle_dir / "performance_layers.csv").open("w", newline="", encoding="utf-8") as f:
            rows = telemetry.get("layers", [])
            fields = sorted({k for row in rows for k in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    if benchmark_dir.exists():
        dst = bundle_dir / "benchmark"
        dst.mkdir(exist_ok=True)
        for p in sorted(benchmark_dir.glob("*")):
            if p.is_file():
                shutil.copy2(p, dst / p.name)
        bench_cfg = repo_dir / "benchmarks" / "config.json"
        if bench_cfg.exists():
            shutil.copy2(bench_cfg, dst / "config.json")

    status["finished_utc"] = utc_now()
    dump_json(bundle_dir / "run_status.json", status)
    write_source_hashes(repo_dir, bundle_dir / "pcscs_source_hashes.json")
    freeze = run([sys.executable, "-m", "pip", "freeze"], capture=True)
    (bundle_dir / "pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")

    (bundle_dir / "INGEST_ME.txt").write_text(
        "Upload pcscs_validation_bundle.zip to ChatGPT.\n"
        "The bundle contains the frozen sample definition, exact source hashes, scientific PCSCS outputs, "
        "end-to-end performance telemetry, controlled scaling benchmark results, environment metadata, and checksums.\n"
        "Raw GBIF images and multi-gigabyte activation tensors are excluded; the manifest preserves image URLs and SHA-256 hashes.\n",
        encoding="utf-8",
    )
    write_checksums(bundle_dir)

def make_zip(bundle_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(bundle_dir))
    print(f"\nFINAL BUNDLE: {zip_path}", flush=True)
    print(f"SHA-256: {sha256_file(zip_path)}", flush=True)


def maybe_download_in_colab(zip_path: Path, auto_download: bool) -> None:
    if not auto_download:
        return
    try:
        from google.colab import files  # type: ignore
        print("Starting Colab download...", flush=True)
        files.download(str(zip_path))
    except Exception as exc:
        print(f"Automatic Colab download unavailable: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-url", default=REPO_URL_DEFAULT)
    p.add_argument(
        "--ref",
        default="main",
        help="Git ref to validate. The resolved commit is frozen into the output.",
    )
    p.add_argument("--work-root", type=Path, default=WORK_ROOT_DEFAULT)
    p.add_argument("--output", type=Path, default=BUNDLE_DEFAULT)
    p.add_argument("--telemetry-interval", type=float, default=1.0)
    p.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the controlled scaling benchmark. The definitive scientific validation still runs.",
    )
    p.add_argument(
        "--benchmark-sample-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Optional benchmark-size override for smoke testing.",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not automatically download the ZIP when running in Colab.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    work_root: Path = args.work_root
    bundle_dir = work_root / "bundle"
    results_dir = work_root / "results"
    benchmark_dir = work_root / "benchmark_results"
    scratch_dir = work_root / "scratch"

    status: dict[str, Any] = {
        "schema_version": 2,
        "validation_name": "PCSCS definitive empirical validation",
        "started_utc": utc_now(),
        "success": False,
        "seed": SEED,
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "stages": [],
    }

    repo_dir: Path | None = None
    config_path: Path | None = None
    data_dir: Path | None = None
    monitor = None
    telemetry = None

    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if work_root.exists():
            shutil.rmtree(work_root)
        t0 = __import__("time").perf_counter()
        repo_dir = bootstrap_repo(args.repo_url, args.ref, work_root, status)
        status["stages"].append({"stage": "bootstrap_repo", "ok": True, "duration_s": __import__("time").perf_counter() - t0})

        from pcscs.performance import PerformanceMonitor
        monitor = PerformanceMonitor(interval_s=args.telemetry_interval).start()

        config_path = repo_dir / "experiments" / "empirical_validation" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        cfg = load_json(config_path)
        data_dir = work_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        with monitor.phase("candidate_pool_query"):
            candidate_pool_path = build_candidate_pool(cfg, config_path, data_dir)
        status["stages"].append({"stage": "build_candidate_pool", "ok": True})

        with monitor.phase("sample_freeze_and_download"):
            manifest_path, image_dir = freeze_sample(cfg, config_path, candidate_pool_path, data_dir)
        status["stages"].append({"stage": "freeze_sample", "ok": True})

        with monitor.phase("sample_validation"):
            validate_sample(cfg, config_path, manifest_path, image_dir)
        status["stages"].append({"stage": "validate_sample", "ok": True})

        with monitor.phase("scientific_validation"):
            run_pcscs_validation(
                cfg, config_path, manifest_path, image_dir, results_dir, scratch_dir, status, monitor
            )
        status["stages"].append({"stage": "run_pcscs_validation", "ok": True})

        with monitor.phase("result_validation"):
            validate_results(cfg, config_path, manifest_path, results_dir)
        status["stages"].append({"stage": "validate_results", "ok": True})

        if not args.skip_benchmark:
            cmd = [
                sys.executable,
                str(repo_dir / "benchmarks" / "benchmark_pcscs.py"),
                "--config", str(repo_dir / "benchmarks" / "config.json"),
                "--output-dir", str(benchmark_dir),
            ]
            if args.benchmark_sample_sizes:
                cmd += ["--sample-sizes", *[str(x) for x in args.benchmark_sample_sizes]]
            with monitor.phase("controlled_scaling_benchmark"):
                run(cmd, cwd=repo_dir)
            status["stages"].append({"stage": "controlled_scaling_benchmark", "ok": True})

        status["success"] = True

    except Exception as exc:
        status["success"] = False
        status["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print("\nVALIDATION FAILED", file=sys.stderr)
        traceback.print_exc()

    finally:
        try:
            if monitor is not None:
                monitor.stop()
                telemetry = monitor.to_dict()
                if data_dir is not None and (data_dir / "sample-manifest.json").exists():
                    n = len(load_json(data_dir / "sample-manifest.json")["samples"])
                    wall = telemetry.get("summary", {}).get("wall_clock_s")
                    if wall:
                        telemetry["summary"]["samples_per_second_end_to_end"] = float(n / wall)

            if repo_dir is not None and config_path is not None and data_dir is not None:
                collect_bundle(
                    repo_dir,
                    config_path,
                    data_dir,
                    results_dir,
                    benchmark_dir,
                    bundle_dir,
                    status,
                    telemetry,
                )
            else:
                bundle_dir.mkdir(parents=True, exist_ok=True)
                status["finished_utc"] = utc_now()
                dump_json(bundle_dir / "run_status.json", status)
                if telemetry is not None:
                    dump_json(bundle_dir / "performance_telemetry.json", telemetry)
                (bundle_dir / "INGEST_ME.txt").write_text(
                    "Upload this bundle to ChatGPT. The validation failed before repository/data setup completed; "
                    "run_status.json contains the diagnostic traceback.\n",
                    encoding="utf-8",
                )
                write_checksums(bundle_dir)

            make_zip(bundle_dir, args.output)
            maybe_download_in_colab(args.output, False if os.environ.get("COLAB_RELEASE_TAG") else (not args.no_download))
        except Exception:
            traceback.print_exc()

    if not status["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
