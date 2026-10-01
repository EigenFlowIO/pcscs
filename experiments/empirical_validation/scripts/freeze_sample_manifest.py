from __future__ import annotations
import datetime as dt
import io
from pathlib import Path
import requests
from PIL import Image
from common import CONFIG_PATH, CANDIDATE_POOL_PATH, MANIFEST_PATH, IMAGE_DIR, load_json, dump_json, sha256_bytes, config_hash

USER_AGENT = "PCSCS-research-reproduction/1.0"


def suffix_for(content_type: str | None, url: str) -> str:
    ct = (content_type or "").lower()
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "tiff" in ct: return ".tif"
    low = url.lower().split("?")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"):
        if low.endswith(ext): return ext
    return ".jpg"


def main():
    cfg = load_json(CONFIG_PATH)
    pool = load_json(CANDIDATE_POOL_PATH)
    if pool.get("config_sha256") != config_hash():
        raise RuntimeError("Candidate pool was built with a different config.json")
    target = int(cfg["target_per_family"])
    min_w, min_h = int(cfg["minimum_width"]), int(cfg["minimum_height"])
    session = requests.Session(); session.headers.update({"User-Agent": USER_AGENT})
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    selected = []
    for fam in pool["families"]:
        family = fam["family"]
        accepted = 0
        for cand in sorted(fam["candidates"], key=lambda x: (x["occurrence_key"], x["media_url"])):
            if accepted >= target: break
            try:
                r = session.get(cand["media_url"], timeout=45)
                r.raise_for_status()
                raw = r.content
                im = Image.open(io.BytesIO(raw)); im.verify()
                im = Image.open(io.BytesIO(raw))
                width, height = im.size
                if width < min_w or height < min_h:
                    continue
                sha = sha256_bytes(raw)
                ext = suffix_for(r.headers.get("Content-Type"), cand["media_url"])
                filename = f"{family}__{cand['occurrence_key']}__{sha[:12]}{ext}"
                path = IMAGE_DIR / filename
                path.write_bytes(raw)
                selected.append({
                    "sample_index": len(selected),
                    "family": family,
                    "imagenet_class": fam["imagenet_class"],
                    "imagenet_index": fam["imagenet_index"],
                    **cand,
                    "filename": filename,
                    "sha256": sha,
                    "width": width,
                    "height": height
                })
                accepted += 1
                print(f"{family}: accepted {accepted}/{target} occurrence {cand['occurrence_key']}")
            except Exception as exc:
                print(f"{family}: rejected occurrence {cand.get('occurrence_key')}: {exc}")
        if accepted != target:
            raise RuntimeError(f"{family}: only {accepted}/{target} valid specimens; expand/fix candidate pool rather than silently changing design")
    manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_sha256": config_hash(),
        "candidate_pool_generated_utc": pool.get("generated_utc"),
        "selection_rule": "For each family, sort archived candidates by (occurrence_key, media_url), then accept the first target_per_family images that download, decode, and meet the fixed minimum dimensions.",
        "target_total": cfg["target_total"],
        "target_per_family": target,
        "samples": selected
    }
    if len(selected) != int(cfg["target_total"]):
        raise RuntimeError(f"Expected {cfg['target_total']} samples, got {len(selected)}")
    dump_json(MANIFEST_PATH, manifest)
    print(f"Frozen {len(selected)} samples in {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
