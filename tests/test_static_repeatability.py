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


def test_static_dataset_directory_is_repository_target():
    readme = (STATIC_DIR / "README.md").read_text(encoding="utf-8")
    assert "exact **250 image files**" in readme
    assert "static_dataset/images" in readme or "images/" in readme
    assert "MIT license" in readme


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


def test_static_repeatability_has_no_sample_acquisition_or_url_fallback():
    source = (ROOT / "experiments" / "empirical_validation" / "static_repeatability.py").read_text(encoding="utf-8")
    assert "build_candidate_pool(" not in source
    assert "freeze_sample(" not in source
    assert "import cache_static_dataset" not in source
    assert "gbif_cache_url" not in source

def test_static_repeatability_initializes_shared_runtime_status():
    source = (ROOT / "experiments" / "empirical_validation" / "static_repeatability.py").read_text(encoding="utf-8")
    assert '"runtime": {"python": sys.version' in source
