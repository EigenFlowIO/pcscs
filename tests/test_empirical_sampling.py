from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "empirical_validation" / "pcscs_colab_validation.py"
spec = importlib.util.spec_from_file_location("pcscs_colab_validation", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_still_image_urls_uses_identifiers_not_reference_pages():
    rec = {
        "media": [
            {"type": "StillImage", "identifier": "https://example.org/a.jpg", "references": "https://example.org/a-page"},
            {"type": "StillImage", "identifier": "https://example.org/a.jpg"},
            {"type": "StillImage", "identifier": "https://example.org/b.jpg"},
            {"type": "Sound", "identifier": "https://example.org/nope.mp3"},
        ]
    }
    assert module.still_image_urls(rec) == [
        "https://example.org/a.jpg",
        "https://example.org/b.jpg",
    ]
    assert module.still_image_references(rec) == ["https://example.org/a-page"]


def test_media_license_normalization_accepts_only_permissive_defaults():
    assert module.normalize_media_license("https://creativecommons.org/publicdomain/zero/1.0/") == "CC0"
    assert module.normalize_media_license("CC0 1.0") == "CC0"
    assert module.normalize_media_license("https://creativecommons.org/licenses/by/4.0/") == "CC-BY"
    assert module.normalize_media_license("CC BY 4.0") == "CC-BY"
    assert module.normalize_media_license("https://creativecommons.org/licenses/by-nc/4.0/") is None
    assert module.normalize_media_license("CC BY-SA 4.0") is None
    assert module.normalize_media_license("") is None


def test_still_image_records_filters_by_exact_media_license():
    rec = {
        "media": [
            {
                "type": "StillImage",
                "identifier": "https://example.org/cc0.jpg",
                "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                "creator": "A. Example",
            },
            {
                "type": "StillImage",
                "identifier": "https://example.org/by.jpg",
                "license": "CC BY 4.0",
                "rightsHolder": "Example Museum",
            },
            {
                "type": "StillImage",
                "identifier": "https://example.org/nc.jpg",
                "license": "CC BY-NC 4.0",
            },
            {
                "type": "StillImage",
                "identifier": "https://example.org/unknown.jpg",
            },
        ]
    }
    rows = module.still_image_records(rec, {"CC0", "CC-BY"})
    assert [x["identifier"] for x in rows] == [
        "https://example.org/cc0.jpg",
        "https://example.org/by.jpg",
    ]
    assert rows[0]["normalized_license"] == "CC0"
    assert rows[1]["normalized_license"] == "CC-BY"


def test_gbif_occurrence_cache_url_matches_documented_addressing():
    identifier = "https://example.org/specimen.jpg"
    expected_md5 = hashlib.md5(identifier.encode("utf-8")).hexdigest()
    assert module.gbif_occurrence_cache_url(123, identifier) == (
        f"https://api.gbif.org/v1/image/cache/occurrence/123/media/{expected_md5}"
    )


def test_validate_sample_allows_unequal_family_counts(tmp_path: Path):
    config = {
        "target_total": 3,
        "minimum_width": 2,
        "minimum_height": 2,
        "families": [
            {"family": "FamilyA"},
            {"family": "FamilyB"},
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha = module.sha256_file(config_path)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    samples = []
    for i, family in enumerate(["FamilyA", "FamilyA", "FamilyB"]):
        name = f"{i}.png"
        path = image_dir / name
        Image.new("RGB", (3, 3), (i, i, i)).save(path)
        samples.append({
            "sample_index": i,
            "occurrence_key": i + 1,
            "family": family,
            "media_license_normalized": "CC0",
            "filename": name,
            "sha256": _sha(path),
            "width": 3,
            "height": 3,
        })

    manifest_path = tmp_path / "sample-manifest.json"
    manifest_path.write_text(json.dumps({
        "config_sha256": config_sha,
        "samples": samples,
    }), encoding="utf-8")

    module.validate_sample(config, config_path, manifest_path, image_dir)
