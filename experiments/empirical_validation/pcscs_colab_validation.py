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
10. writes one compact results ZIP designed for direct ingestion by ChatGPT, and
11. writes a second dataset ZIP containing the exact accepted specimen images.

Candidate media are restricted to explicitly redistributable licenses configured
in config.json. The default allowlist accepts only CC0 and CC BY media. The image
bytes are preserved in a separate dataset artifact so repeatability does not depend
on future availability of GBIF or publisher image servers.

Recommended Colab invocation:
    !python /content/pcscs/experiments/empirical_validation/pcscs_colab_validation.py

Optional:
    !python /content/pcscs/experiments/empirical_validation/pcscs_colab_validation.py --ref <git-ref>

The final artifact is:
    /content/pcscs_validation_bundle.zip
and the frozen image dataset is:
    /content/pcscs_validation_dataset.zip
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
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

REPO_URL_DEFAULT = "https://github.com/EigenFlowIO/pcscs.git"
WORK_ROOT_DEFAULT = Path("/content/pcscs_validation_work")
BUNDLE_DEFAULT = Path("/content/pcscs_validation_bundle.zip")
DATASET_BUNDLE_DEFAULT = Path("/content/pcscs_validation_dataset.zip")
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


def still_image_urls(record: dict[str, Any]) -> list[str]:
    """Return unique publisher media identifiers for StillImage records.

    GBIF distinguishes the direct media ``identifier`` from ``references``
    (typically an HTML/resource page). Only identifiers are valid inputs to
    GBIF's occurrence-image cache.
    """
    urls: list[str] = []
    for media in record.get("media", []):
        if media.get("type") != "StillImage":
            continue
        url = media.get("identifier")
        if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in urls:
            urls.append(url)
    return urls


def normalize_media_license(value: Any) -> str | None:
    """Normalize clearly redistributable Creative Commons media licenses.

    GBIF multimedia licenses are publisher-supplied strings. We intentionally
    recognize only licenses whose redistribution terms are unambiguous for a
    public reproducibility dataset. Non-commercial, share-alike, all-rights-
    reserved, blank, and unknown values are rejected by default.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    low = value.strip().lower().replace("https://", "http://")
    compact = " ".join(low.replace("_", " ").replace("-", " ").split())

    if "creativecommons.org/publicdomain/zero/" in low or compact in {
        "cc0", "cc0 1.0", "creative commons zero", "creative commons zero 1.0"
    }:
        return "CC0"

    # Reject restrictive variants before accepting generic attribution text.
    if any(token in low for token in ("/by-nc/", "/by-sa/", "/by-nd/", "/by-nc-sa/", "/by-nc-nd/")):
        return None
    if any(token in compact for token in ("cc by nc", "cc by sa", "cc by nd")):
        return None
    if "creativecommons.org/licenses/by/" in low or compact in {
        "cc by", "cc by 2.0", "cc by 2.5", "cc by 3.0", "cc by 4.0",
        "creative commons attribution", "creative commons attribution 4.0",
    }:
        return "CC-BY"
    return None


def build_media_attribution(media: dict[str, Any], occurrence_key: int, institution_code: str | None = None) -> str:
    """Build a conservative human-readable attribution string from supplied media metadata."""
    creator = str(media.get("creator") or "").strip()
    rights_holder = str(media.get("rightsHolder") or "").strip()
    publisher = str(media.get("publisher") or "").strip()
    source = str(media.get("identifier") or "").strip()
    license_text = str(media.get("license") or "").strip()
    credit = creator or rights_holder or publisher or (institution_code or "").strip() or "Source provider"
    parts = [credit]
    if rights_holder and rights_holder != credit:
        parts.append(rights_holder)
    if institution_code and institution_code not in parts:
        parts.append(institution_code)
    parts.append(f"GBIF occurrence {int(occurrence_key)}")
    if license_text:
        parts.append(license_text)
    if source:
        parts.append(source)
    return " | ".join(parts)


def still_image_records(record: dict[str, Any], allowlist: set[str]) -> list[dict[str, Any]]:
    """Return StillImage media records whose exact media license is allowed."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for media in record.get("media", []):
        if media.get("type") != "StillImage":
            continue
        identifier = media.get("identifier")
        if not isinstance(identifier, str) or not identifier.startswith(("http://", "https://")):
            continue
        normalized = normalize_media_license(media.get("license"))
        if normalized not in allowlist or identifier in seen:
            continue
        seen.add(identifier)
        rows.append({
            "identifier": identifier,
            "license": media.get("license"),
            "normalized_license": normalized,
            "creator": media.get("creator"),
            "rights_holder": media.get("rightsHolder"),
            "publisher": media.get("publisher"),
            "references": media.get("references"),
            "title": media.get("title"),
        })
    return rows


def still_image_references(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for media in record.get("media", []):
        if media.get("type") != "StillImage":
            continue
        url = media.get("references")
        if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in refs:
            refs.append(url)
    return refs


def gbif_occurrence_cache_url(occurrence_key: int, media_identifier: str) -> str:
    """Build GBIF's documented cached occurrence-image URL.

    GBIF identifies a media item in its cache by the occurrence key plus the
    hexadecimal MD5 digest of the publisher's media identifier URL. MD5 here is
    an addressing convention defined by the GBIF API, not a security primitive.
    """
    digest = hashlib.md5(media_identifier.encode("utf-8")).hexdigest()
    return (
        "https://api.gbif.org/v1/image/cache/occurrence/"
        f"{int(occurrence_key)}/media/{digest}"
    )


def _download_image_from_gbif_cache(
    session: Any,
    occurrence_key: int,
    identifiers: list[str],
    timeout: tuple[int, int] = (8, 60),
) -> tuple[bytes, str, str, str | None, list[dict[str, Any]]]:
    """Retrieve an occurrence image through GBIF's image cache.

    Publisher image servers are intentionally *not* scraped directly. GBIF's
    own documentation provides a cache endpoint for occurrence media and asks
    scripted clients to limit usage to a single HTTP connection. The caller
    therefore supplies one shared ``requests.Session`` and selection is
    sequential.

    Returns raw bytes, cache URL, publisher identifier, content type, and an
    attempt log.
    """
    attempts: list[dict[str, Any]] = []
    for identifier in identifiers:
        cache_url = gbif_occurrence_cache_url(occurrence_key, identifier)
        try:
            r = session.get(cache_url, timeout=timeout, allow_redirects=True)
            attempts.append({
                "identifier": identifier,
                "cache_url": cache_url,
                "status": int(r.status_code),
            })
            if r.status_code == 200 and r.content:
                return r.content, cache_url, identifier, r.headers.get("Content-Type"), attempts
        except Exception as exc:
            attempts.append({
                "identifier": identifier,
                "cache_url": cache_url,
                "error": type(exc).__name__,
                "message": str(exc)[:240],
            })
    raise RuntimeError(f"all GBIF cache URLs failed ({len(attempts)} HTTP attempts)")


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
    allowlist = set(cfg.get("media_license_allowlist", ["CC0", "CC-BY"]))
    for fam in cfg["families"]:
        family = fam["family"]
        m = session.get(gbif_match, params={"name": family}, timeout=30)
        m.raise_for_status()
        usage_key = m.json().get("usageKey")
        if not usage_key:
            raise RuntimeError(f"No GBIF usageKey for {family}")

        scan_limit = int(cfg.get("candidate_scan_limit_per_family", limit))
        candidates = []
        seen_occurrences: set[int] = set()
        scanned = 0
        offset = 0
        page_size = min(300, scan_limit)
        while scanned < scan_limit and len(candidates) < limit:
            params = {
                "taxonKey": usage_key,
                "mediaType": cfg["gbif"]["mediaType"],
                "basisOfRecord": cfg["gbif"]["basisOfRecord"],
                "limit": min(page_size, scan_limit - scanned),
                "offset": offset,
            }
            r = session.get(gbif_search, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
            results = payload.get("results", [])
            if not results:
                break
            scanned += len(results)
            offset += len(results)

            for rec in results:
                media_rows = still_image_records(rec, allowlist)
                refs = still_image_references(rec)
                key = rec.get("key")
                if key is None or not media_rows:
                    continue
                key = int(key)
                if key in seen_occurrences:
                    continue
                seen_occurrences.add(key)
                for media_row in media_rows:
                    media_row["attribution_text"] = build_media_attribution(
                        {
                            "creator": media_row.get("creator"),
                            "rightsHolder": media_row.get("rights_holder"),
                            "publisher": media_row.get("publisher"),
                            "identifier": media_row.get("identifier"),
                            "license": media_row.get("license"),
                        },
                        key,
                        rec.get("institutionCode"),
                    )
                urls = [m["identifier"] for m in media_rows]
                candidates.append(
                    {
                        "occurrence_key": key,
                        "media_url": urls[0],
                        "media_identifiers": urls,
                        "media_records": media_rows,
                        "media_references": refs,
                        "scientific_name": rec.get("scientificName"),
                        "species": rec.get("species"),
                        "institution_code": rec.get("institutionCode"),
                        "collection_code": rec.get("collectionCode"),
                        "catalog_number": rec.get("catalogNumber"),
                    }
                )
                if len(candidates) >= limit:
                    break

            if payload.get("endOfRecords"):
                break

        candidates.sort(key=lambda x: (x["occurrence_key"], x["media_url"]))
        out["families"].append({**fam, "gbif_usage_key": usage_key, "candidates": candidates})
        print(
            f"{family}: {len(candidates)} candidates with allowed media licenses "
            f"({', '.join(sorted(allowlist))})",
            flush=True,
        )

    candidate_pool_path = data_dir / "candidate-pool.json"
    dump_json(candidate_pool_path, out)
    return candidate_pool_path


def _find_next_valid_candidate(
    fam: dict[str, Any],
    start_index: int,
    min_w: int,
    min_h: int,
    session: Any,
    timeout: tuple[int, int],
) -> tuple[dict[str, Any] | None, int, dict[str, Any]]:
    """Scan one family's candidates in deterministic order until one image is usable."""
    from PIL import Image

    family = fam["family"]
    candidates = sorted(
        fam["candidates"],
        key=lambda x: (x["occurrence_key"], x.get("media_url", "")),
    )
    diag = {
        "family": family,
        "candidate_attempts": 0,
        "http_attempts": 0,
        "failures": {},
    }

    for idx in range(start_index, len(candidates)):
        cand = candidates[idx]
        diag["candidate_attempts"] += 1
        identifiers = list(cand.get("media_identifiers") or [cand.get("media_url")])
        identifiers = [u for u in identifiers if isinstance(u, str) and u.startswith(("http://", "https://"))]
        try:
            raw, cache_url, successful_identifier, content_type, attempts = _download_image_from_gbif_cache(
                session,
                int(cand["occurrence_key"]),
                identifiers,
                timeout=timeout,
            )
            diag["http_attempts"] += len(attempts)
            for a in attempts:
                if "status" in a and a["status"] != 200:
                    key = f"http_{a['status']}"
                    diag["failures"][key] = diag["failures"].get(key, 0) + 1

            im = Image.open(io.BytesIO(raw))
            im.verify()
            im = Image.open(io.BytesIO(raw))
            width, height = im.size
            if width < min_w or height < min_h:
                diag["failures"]["below_minimum_dimensions"] = diag["failures"].get("below_minimum_dimensions", 0) + 1
                continue

            media_record = next(
                (m for m in cand.get("media_records", []) if m.get("identifier") == successful_identifier),
                {},
            )

            return ({
                "family": family,
                "imagenet_class": fam["imagenet_class"],
                "imagenet_index": fam["imagenet_index"],
                **cand,
                "media_url": successful_identifier,
                "publisher_media_identifier": successful_identifier,
                "gbif_cache_url": cache_url,
                "original_media_url": cand.get("media_url"),
                "media_license": media_record.get("license"),
                "media_license_normalized": media_record.get("normalized_license"),
                "media_creator": media_record.get("creator"),
                "media_rights_holder": media_record.get("rights_holder"),
                "media_publisher": media_record.get("publisher"),
                "media_reference": media_record.get("references"),
                "media_attribution": media_record.get("attribution_text"),
                "downloaded_content_type": content_type,
                "raw_bytes": raw,
                "width": width,
                "height": height,
            }, idx + 1, diag)
        except Exception as exc:
            msg = str(exc)
            diag["failures"]["download_or_decode"] = diag["failures"].get("download_or_decode", 0) + 1
            print(
                f"{family}: rejected occurrence {cand.get('occurrence_key')}: {msg}",
                flush=True,
            )

    return None, len(candidates), diag


def freeze_sample(
    cfg: dict[str, Any],
    config_path: Path,
    candidate_pool_path: Path,
    data_dir: Path,
) -> tuple[Path, Path]:
    from collections import Counter

    pool = load_json(candidate_pool_path)
    config_sha = sha256_file(config_path)
    if pool.get("config_sha256") != config_sha:
        raise RuntimeError("Candidate pool was built with a different config.json")

    target_total = int(cfg["target_total"])
    min_w = int(cfg["minimum_width"])
    min_h = int(cfg["minimum_height"])
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # The family list defines sampling strata, not hard quotas. Selection is a
    # deterministic round-robin over the configured family order. Each active
    # family contributes at most one specimen per round; exhausted/inaccessible
    # families are skipped and the remaining families absorb the deficit.
    families = list(pool["families"])
    next_index = {fam["family"]: 0 for fam in families}
    exhausted = {fam["family"]: False for fam in families}
    selected: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "selection_design": "deterministic_adaptive_round_robin_family_stratified",
        "image_acquisition": "gbif_occurrence_image_cache_single_connection",
        "target_total": target_total,
        "families": {
            fam["family"]: {
                "candidate_attempts": 0,
                "http_attempts": 0,
                "accepted": 0,
                "failures": {},
            }
            for fam in families
        },
    }

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    connect_timeout = int(cfg.get("download", {}).get("connect_timeout_s", 8))
    read_timeout = int(cfg.get("download", {}).get("read_timeout_s", 60))
    download_timeout = (connect_timeout, read_timeout)
    cache_session = requests.Session()
    cache_session.headers.update({"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"})
    cache_session.mount(
        "https://",
        HTTPAdapter(max_retries=Retry(
            total=1, connect=1, read=1, status=1, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]), raise_on_status=False,
        )),
    )

    round_number = 0
    while len(selected) < target_total:
        active = [fam for fam in families if not exhausted[fam["family"]]]
        if not active:
            break
        round_number += 1
        print(
            f"selection round {round_number}: {len(selected)}/{target_total} accepted; "
            f"{len(active)} active families",
            flush=True,
        )

        # GBIF explicitly asks scripted image-cache users to limit usage to a
        # single HTTP connection. One shared session is therefore used and
        # families are processed sequentially in configured order. This is both
        # gentler on the service and deterministic.
        for fam in active:
            if len(selected) >= target_total:
                break
            family = fam["family"]
            result, new_index, diag = _find_next_valid_candidate(
                fam,
                next_index[family],
                min_w,
                min_h,
                cache_session,
                download_timeout,
            )
            next_index[family] = new_index
            agg = diagnostics["families"][family]
            agg["candidate_attempts"] += diag["candidate_attempts"]
            agg["http_attempts"] += diag["http_attempts"]
            for key, value in diag["failures"].items():
                agg["failures"][key] = agg["failures"].get(key, 0) + value

            if result is None:
                exhausted[family] = True
                continue

            raw = result.pop("raw_bytes")
            sha = sha256_bytes(raw)
            ext = suffix_for(result.get("downloaded_content_type"), result["media_url"])
            filename = f"{family}__{result['occurrence_key']}__{sha[:12]}{ext}"
            (image_dir / filename).write_bytes(raw)
            result.update({
                "sample_index": len(selected),
                "filename": filename,
                "sha256": sha,
            })
            selected.append(result)
            agg["accepted"] += 1
            print(
                f"{family}: accepted occurrence {result['occurrence_key']} "
                f"({len(selected)}/{target_total} total)",
                flush=True,
            )

    cache_session.close()

    family_counts = Counter(x["family"] for x in selected)
    diagnostics["rounds"] = round_number
    diagnostics["selected_total"] = len(selected)
    diagnostics["family_counts"] = dict(sorted(family_counts.items()))
    diagnostics["exhausted_families"] = sorted(k for k, v in exhausted.items() if v)
    dump_json(data_dir / "sample-selection-diagnostics.json", diagnostics)

    if len(selected) != target_total:
        raise RuntimeError(
            f"Could obtain only {len(selected)}/{target_total} valid specimens across the full "
            f"configured candidate pool. Family availability: {dict(sorted(family_counts.items()))}"
        )

    manifest = {
        "schema_version": 2,
        "experiment_id": cfg["experiment_id"],
        "frozen_utc": utc_now(),
        "config_sha256": config_sha,
        "candidate_pool_generated_utc": pool.get("generated_utc"),
        "selection_rule": (
            "Deterministic adaptive round-robin across configured family strata. "
            "Only StillImage media with an exact license normalized to the configured "
            "media_license_allowlist are eligible. "
            "Families are visited in config order; each active family contributes at most one "
            "valid specimen per round. Families that exhaust accessible candidates are skipped, "
            "and remaining families continue until target_total is reached. No per-family quota is enforced."
        ),
        "target_total": target_total,
        "media_license_allowlist": cfg.get("media_license_allowlist", ["CC0", "CC-BY"]),
        "family_counts": dict(sorted(family_counts.items())),
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
    unexpected = set(counts) - expected_fams
    if unexpected:
        errors.append(f"unexpected families: {sorted(unexpected)}")

    allowed_licenses = set(cfg.get("media_license_allowlist", ["CC0", "CC-BY"]))

    for row in samples:
        if row.get("media_license_normalized") not in allowed_licenses:
            errors.append(
                f"disallowed or missing media license {row.get('media_license_normalized')!r} "
                f"for {row.get('filename')}"
            )
        if not row.get("media_license"):
            errors.append(f"missing original media license for {row.get('filename')}")
        if not row.get("publisher_media_identifier"):
            errors.append(f"missing publisher media identifier for {row.get('filename')}")
        if not row.get("media_attribution"):
            errors.append(f"missing attribution text for {row.get('filename')}")
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
        "media_license",
        "media_license_normalized",
        "media_creator",
        "media_rights_holder",
        "media_publisher",
        "media_reference",
        "media_attribution",
        "publisher_media_identifier",
        "original_media_url",
        "gbif_cache_url",
        "filename",
        "sha256",
        "width",
        "height",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_image_licenses_csv(manifest_path: Path, out_path: Path) -> None:
    rows = load_json(manifest_path)["samples"]
    fields = [
        "sample_index", "filename", "sha256", "occurrence_key", "family",
        "scientific_name", "institution_code", "collection_code", "catalog_number",
        "media_license", "media_license_normalized", "media_creator",
        "media_rights_holder", "media_publisher", "media_reference",
        "media_attribution", "original_media_url", "gbif_cache_url",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_dataset_zip(
    config_path: Path,
    manifest_path: Path,
    image_dir: Path,
    zip_path: Path,
) -> dict[str, Any]:
    """Create a self-contained frozen image dataset for static repeatability."""
    tmp = zip_path.parent / "pcscs_validation_dataset_work"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    images_out = tmp / "images"
    images_out.mkdir()

    shutil.copy2(config_path, tmp / "config.json")
    shutil.copy2(manifest_path, tmp / "sample-manifest.json")
    write_manifest_csv(manifest_path, tmp / "sample_manifest.csv")
    write_image_licenses_csv(manifest_path, tmp / "IMAGE_LICENSES.csv")

    manifest = load_json(manifest_path)
    for row in manifest["samples"]:
        src = image_dir / row["filename"]
        if not src.exists() or sha256_file(src) != row["sha256"]:
            raise RuntimeError(f"Cannot freeze dataset: image mismatch {row['filename']}")
        shutil.copy2(src, images_out / row["filename"])

    metadata = {
        "schema_version": 1,
        "dataset_id": f"{manifest.get('experiment_id', 'pcscs-validation')}-static-images",
        "created_utc": utc_now(),
        "sample_count": len(manifest["samples"]),
        "manifest_sha256": sha256_file(manifest_path),
        "license_policy": (
            "Every included image was selected only when GBIF supplied an exact multimedia "
            "license normalized to CC0 or CC-BY. IMAGE_LICENSES.csv preserves the original "
            "license and attribution metadata. Third-party image licenses are separate from "
            "the PCSCS software license."
        ),
    }
    dump_json(tmp / "dataset-metadata.json", metadata)
    (tmp / "README.txt").write_text(
        "PCSCS frozen empirical image dataset.\n"
        "These are the exact image bytes used by the validation run.\n"
        "Verify checksums before analysis. Image licenses are recorded in IMAGE_LICENSES.csv.\n",
        encoding="utf-8",
    )
    write_checksums(tmp)
    make_zip(tmp, zip_path)
    result = {
        "path": str(zip_path),
        "sha256": sha256_file(zip_path),
        "sample_count": len(manifest["samples"]),
        "manifest_sha256": sha256_file(manifest_path),
    }
    shutil.rmtree(tmp)
    return result


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
        (data_dir / "sample-selection-diagnostics.json", "sample-selection-diagnostics.json"),
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
        "Raw images are stored separately in pcscs_validation_dataset.zip; multi-gigabyte activation tensors are excluded.\n",
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
    p.add_argument("--dataset-output", type=Path, default=DATASET_BUNDLE_DEFAULT)
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

        with monitor.phase("freeze_static_dataset"):
            dataset_artifact = make_dataset_zip(
                config_path,
                manifest_path,
                image_dir,
                args.dataset_output,
            )
        status["dataset_artifact"] = dataset_artifact
        status["stages"].append({"stage": "freeze_static_dataset", "ok": True})

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
