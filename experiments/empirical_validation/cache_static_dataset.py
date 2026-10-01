#!/usr/bin/env python3
"""Materialize the frozen PCSCS validation image set from its exact manifest.

This is a one-time dataset-freeze utility. It does not select a new sample.
It downloads only the GBIF cache URLs already recorded in the definitive
sample manifest and accepts a file only when its SHA-256 and dimensions match
that manifest exactly.

Once every image has been materialized and verified, the image directory can
be committed alongside the manifest. Future repeatability runs should use
``static_repeatability.py`` and make no GBIF image requests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

USER_AGENT = "PCSCS-static-dataset-freeze/1.0 (+https://github.com/EigenFlowIO/pcscs)"
HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "static_dataset"


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


def verify_image_bytes(raw: bytes, row: dict[str, Any]) -> None:
    from PIL import Image

    actual = sha256_bytes(raw)
    if actual != row["sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch for {row['filename']}: expected {row['sha256']}, got {actual}"
        )
    with Image.open(io.BytesIO(raw)) as im:
        size = (int(im.width), int(im.height))
    expected = (int(row["width"]), int(row["height"]))
    if size != expected:
        raise RuntimeError(
            f"dimension mismatch for {row['filename']}: expected {expected}, got {size}"
        )


def verify_materialized_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_dir / "sample-manifest.json"
    images_dir = dataset_dir / "images"
    manifest = load_json(manifest_path)
    samples = sorted(manifest["samples"], key=lambda r: int(r["sample_index"]))
    errors: list[str] = []
    rows: list[dict[str, Any]] = []

    for row in samples:
        path = images_dir / row["filename"]
        status = "ok"
        message = ""
        if not path.exists():
            status = "missing"
            message = "file not present"
            errors.append(f"missing {path.name}")
        else:
            try:
                raw = path.read_bytes()
                verify_image_bytes(raw, row)
            except Exception as exc:
                status = "invalid"
                message = str(exc)
                errors.append(f"{path.name}: {exc}")
        rows.append({
            "sample_index": row["sample_index"],
            "occurrence_key": row["occurrence_key"],
            "family": row["family"],
            "filename": row["filename"],
            "sha256": row["sha256"],
            "status": status,
            "message": message,
        })

    if errors:
        preview = "\n".join(errors[:12])
        more = "" if len(errors) <= 12 else f"\n... plus {len(errors)-12} more"
        raise RuntimeError(f"static dataset verification failed ({len(errors)} errors):\n{preview}{more}")
    return rows


def write_dataset_checksums(dataset_dir: Path) -> str:
    targets = [dataset_dir / "sample-manifest.json"] + sorted(
        p for p in (dataset_dir / "images").iterdir() if p.is_file() and p.name != ".gitkeep"
    )
    lines = []
    for path in targets:
        rel = path.relative_to(dataset_dir).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    body = "\n".join(lines) + "\n"
    (dataset_dir / "checksums.sha256").write_text(body, encoding="utf-8")
    dataset_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return dataset_hash


def fetch_json(session: Any, url: str, timeout: tuple[int, int]) -> dict[str, Any] | None:
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 200:
            value = r.json()
            return value if isinstance(value, dict) else None
    except Exception:
        return None
    return None


def extract_rights_metadata(occurrence: dict[str, Any] | None, publisher_identifier: str) -> dict[str, str]:
    result = {
        "license": "",
        "rights": "",
        "rights_holder": "",
        "creator": "",
        "identifier": publisher_identifier,
    }
    if not occurrence:
        return result

    # Occurrence-level values are useful fallbacks.
    result["license"] = str(occurrence.get("license") or "")
    result["rights"] = str(occurrence.get("rights") or "")
    result["rights_holder"] = str(occurrence.get("rightsHolder") or "")

    for media in occurrence.get("media", []) or []:
        if media.get("identifier") != publisher_identifier:
            continue
        result["license"] = str(media.get("license") or result["license"])
        result["rights"] = str(media.get("rights") or result["rights"])
        result["rights_holder"] = str(media.get("rightsHolder") or result["rights_holder"])
        result["creator"] = str(media.get("creator") or "")
        break
    return result


def materialize(dataset_dir: Path, retries: int, connect_timeout: int, read_timeout: int) -> None:
    import requests

    manifest_path = dataset_dir / "sample-manifest.json"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    samples = sorted(manifest["samples"], key=lambda r: int(r["sample_index"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    timeout = (connect_timeout, read_timeout)
    license_rows: list[dict[str, Any]] = []

    for pos, row in enumerate(samples, start=1):
        path = images_dir / row["filename"]
        if path.exists():
            try:
                verify_image_bytes(path.read_bytes(), row)
                print(f"[{pos:03d}/{len(samples)}] verified existing {path.name}", flush=True)
            except Exception:
                path.unlink()

        if not path.exists():
            url = row.get("gbif_cache_url")
            if not url:
                raise RuntimeError(f"manifest row has no gbif_cache_url: {row['filename']}")
            last_error: Exception | None = None
            raw: bytes | None = None
            for attempt in range(retries + 1):
                try:
                    r = session.get(url, timeout=timeout, allow_redirects=True)
                    if r.status_code != 200 or not r.content:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    raw = r.content
                    verify_image_bytes(raw, row)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < retries:
                        time.sleep(min(2 ** attempt, 5))
            if raw is None:
                raise RuntimeError(f"failed to materialize {row['filename']}: {last_error}")
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(raw)
            os.replace(tmp, path)
            print(f"[{pos:03d}/{len(samples)}] cached {path.name}", flush=True)

        occurrence = fetch_json(
            session,
            f"https://api.gbif.org/v1/occurrence/{int(row['occurrence_key'])}",
            timeout,
        )
        rights = extract_rights_metadata(occurrence, row.get("publisher_media_identifier", ""))
        license_rows.append({
            "sample_index": row["sample_index"],
            "occurrence_key": row["occurrence_key"],
            "family": row["family"],
            "filename": row["filename"],
            "publisher_media_identifier": row.get("publisher_media_identifier", ""),
            "gbif_cache_url": row.get("gbif_cache_url", ""),
            **rights,
        })

    verify_materialized_dataset(dataset_dir)
    dataset_hash = write_dataset_checksums(dataset_dir)

    with (dataset_dir / "IMAGE_LICENSES.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_index", "occurrence_key", "family", "filename",
            "publisher_media_identifier", "gbif_cache_url",
            "license", "rights", "rights_holder", "creator", "identifier",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(license_rows)

    meta_path = dataset_dir / "dataset-metadata.json"
    metadata = load_json(meta_path) if meta_path.exists() else {"schema_version": 1}
    metadata.update({
        "image_state": "materialized_and_hash_verified",
        "sample_count": len(samples),
        "dataset_checksums_sha256": dataset_hash,
        "materialized_utc_epoch": int(time.time()),
        "rights_metadata_file": "IMAGE_LICENSES.csv",
    })
    dump_json(meta_path, metadata)
    print(f"STATIC DATASET VERIFIED: {len(samples)} images", flush=True)
    print(f"DATASET CHECKSUM-MANIFEST SHA-256: {dataset_hash}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--verify-only", action="store_true", help="Verify local bytes only; make no network requests.")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--connect-timeout", type=int, default=8)
    p.add_argument("--read-timeout", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if args.verify_only:
        rows = verify_materialized_dataset(dataset_dir)
        dataset_hash = write_dataset_checksums(dataset_dir)
        print(f"STATIC DATASET VERIFIED: {len(rows)} images")
        print(f"DATASET CHECKSUM-MANIFEST SHA-256: {dataset_hash}")
        return
    materialize(dataset_dir, args.retries, args.connect_timeout, args.read_timeout)


if __name__ == "__main__":
    main()
