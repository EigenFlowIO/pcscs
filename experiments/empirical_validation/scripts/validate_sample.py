from __future__ import annotations
from collections import Counter
from PIL import Image
from common import CONFIG_PATH, MANIFEST_PATH, IMAGE_DIR, load_json, sha256_file, config_hash

def main():
    cfg = load_json(CONFIG_PATH); m = load_json(MANIFEST_PATH)
    errors = []
    if m.get("config_sha256") != config_hash(): errors.append("config hash mismatch")
    samples = m.get("samples", [])
    if len(samples) != int(cfg["target_total"]): errors.append(f"sample count {len(samples)} != {cfg['target_total']}")
    ids = [x["occurrence_key"] for x in samples]
    if len(ids) != len(set(ids)): errors.append("duplicate GBIF occurrence keys")
    counts = Counter(x["family"] for x in samples)
    expected_fams = {f["family"] for f in cfg["families"]}
    if set(counts) != expected_fams: errors.append("family set mismatch")
    for fam in expected_fams:
        if counts[fam] != int(cfg["target_per_family"]): errors.append(f"{fam}: {counts[fam]} samples")
    for row in samples:
        path = IMAGE_DIR / row["filename"]
        if not path.exists(): errors.append(f"missing {row['filename']}"); continue
        if sha256_file(path) != row["sha256"]: errors.append(f"hash mismatch {row['filename']}")
        try:
            with Image.open(path) as im:
                if im.width != row["width"] or im.height != row["height"]: errors.append(f"dimension mismatch {row['filename']}")
                if im.width < cfg["minimum_width"] or im.height < cfg["minimum_height"]: errors.append(f"below minimum size {row['filename']}")
        except Exception as exc: errors.append(f"decode failure {row['filename']}: {exc}")
    if errors:
        raise SystemExit("Sample validation FAILED:\n- " + "\n- ".join(errors))
    print(f"Sample validation PASSED: {len(samples)} specimens; {len(counts)} families; exact hashes verified")

if __name__ == "__main__":
    main()
