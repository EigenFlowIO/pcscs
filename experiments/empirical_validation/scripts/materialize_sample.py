from __future__ import annotations
import requests
from common import MANIFEST_PATH, IMAGE_DIR, load_json, sha256_bytes

USER_AGENT = "PCSCS-research-reproduction/1.0"

def main():
    manifest = load_json(MANIFEST_PATH)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers.update({"User-Agent": USER_AGENT})
    for row in manifest["samples"]:
        path = IMAGE_DIR / row["filename"]
        if path.exists() and sha256_bytes(path.read_bytes()) == row["sha256"]:
            continue
        r = s.get(row["media_url"], timeout=45); r.raise_for_status()
        raw = r.content
        if sha256_bytes(raw) != row["sha256"]:
            raise RuntimeError(f"Remote bytes changed for GBIF occurrence {row['occurrence_key']}; expected SHA-256 {row['sha256']}")
        path.write_bytes(raw)
        print(f"materialized {path.name}")

if __name__ == "__main__":
    main()
