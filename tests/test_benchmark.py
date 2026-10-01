import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "benchmark_pcscs.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("pcscs_benchmark", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_features_are_deterministic():
    mod = load_benchmark_module()
    a = mod.fixed_features(123, 12, 16)
    b = mod.fixed_features(123, 12, 16)
    c = mod.fixed_features(123, 13, 16)
    assert np.array_equal(a, b)
    assert a.shape == (12, 16)
    assert c.shape == (13, 16)


def test_tiny_benchmark_smoke(tmp_path):
    mod = load_benchmark_module()
    cfg = {
        "schema_version": 1,
        "benchmark_id": "TEST",
        "seed": 7,
        "sample_sizes": [6],
        "feature_dimension": 8,
        "n_steps": 12,
        "enable_tracking": False,
        "warmup_runs": 0,
        "measured_repeats": 1,
        "telemetry_interval_s": 0.01,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = mod.run_benchmark(config_path, tmp_path / "out")
    assert "6" in result["conditions"]
    assert (tmp_path / "out" / "benchmark_results.json").exists()
    assert (tmp_path / "out" / "benchmark_runs.csv").exists()
