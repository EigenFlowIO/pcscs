from __future__ import annotations
import datetime as dt
import requests
from common import CONFIG_PATH, CANDIDATE_POOL_PATH, load_json, dump_json, config_hash

GBIF_MATCH = "https://api.gbif.org/v1/species/match"
GBIF_SEARCH = "https://api.gbif.org/v1/occurrence/search"
USER_AGENT = "PCSCS-research-reproduction/1.0"


def first_still_image(record):
    for media in record.get("media", []):
        if media.get("type") == "StillImage" and media.get("identifier"):
            return media["identifier"]
    return None


def main():
    cfg = load_json(CONFIG_PATH)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    out = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_sha256": config_hash(),
        "query_semantics": "Live GBIF provenance snapshot; frozen validation sample is defined by sample-manifest.json, not by re-querying GBIF.",
        "families": []
    }
    limit = int(cfg["candidate_pool_per_family"])
    for fam in cfg["families"]:
        family = fam["family"]
        m = session.get(GBIF_MATCH, params={"name": family}, timeout=30)
        m.raise_for_status()
        usage_key = m.json().get("usageKey")
        if not usage_key:
            raise RuntimeError(f"No GBIF usageKey for {family}")
        params = {
            "taxonKey": usage_key,
            "mediaType": cfg["gbif"]["mediaType"],
            "basisOfRecord": cfg["gbif"]["basisOfRecord"],
            "limit": min(limit, 300)
        }
        r = session.get(GBIF_SEARCH, params=params, timeout=60)
        r.raise_for_status()
        candidates = []
        for rec in r.json().get("results", []):
            url = first_still_image(rec)
            key = rec.get("key")
            if key is None or not url:
                continue
            candidates.append({
                "occurrence_key": int(key),
                "media_url": url,
                "scientific_name": rec.get("scientificName"),
                "species": rec.get("species"),
                "institution_code": rec.get("institutionCode"),
                "collection_code": rec.get("collectionCode"),
                "catalog_number": rec.get("catalogNumber")
            })
        candidates.sort(key=lambda x: (x["occurrence_key"], x["media_url"]))
        out["families"].append({**fam, "gbif_usage_key": usage_key, "candidates": candidates})
        print(f"{family}: {len(candidates)} candidates")
    dump_json(CANDIDATE_POOL_PATH, out)
    print(f"Wrote {CANDIDATE_POOL_PATH}")

if __name__ == "__main__":
    main()
