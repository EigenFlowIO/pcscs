import importlib.util
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE_SCRIPT = ROOT / "experiments" / "empirical_validation" / "cache_static_dataset.py"
STATIC_DIR = ROOT / "experiments" / "empirical_validation" / "static_dataset"


def load_cache_module():
    spec = importlib.util.spec_from_file_location("cache_static_dataset", CACHE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_manifest_defines_180_samples():
    manifest = json.loads((STATIC_DIR / "sample-manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_total"] == 180
    assert len(manifest["samples"]) == 180
    assert len({row["filename"] for row in manifest["samples"]}) == 180
    assert all(len(row["sha256"]) == 64 for row in manifest["samples"])


def test_verify_materialized_dataset_on_tiny_fixture(tmp_path):
    mod = load_cache_module()
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)

    image_path = images / "sample.png"
    Image.new("RGB", (12, 9), (1, 2, 3)).save(image_path)
    sha = mod.sha256_file(image_path)
    manifest = {
        "samples": [{
            "sample_index": 0,
            "occurrence_key": 1,
            "family": "Exampleidae",
            "filename": "sample.png",
            "sha256": sha,
            "width": 12,
            "height": 9,
        }]
    }
    (dataset / "sample-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = mod.verify_materialized_dataset(dataset)
    assert rows[0]["status"] == "ok"
