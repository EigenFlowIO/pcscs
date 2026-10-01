import time

from pcscs.performance import (
    PerformanceMonitor,
    TELEMETRY_SCHEMA_VERSION,
    summarize_samples,
)


def test_performance_monitor_serializes():
    monitor = PerformanceMonitor(interval_s=0.01).start()
    with monitor.phase("unit"):
        time.sleep(0.025)
    monitor.add_layer_metric("layer", flattened_dimension=8)
    monitor.stop()
    doc = monitor.to_dict()
    assert doc["schema_version"] == TELEMETRY_SCHEMA_VERSION
    assert doc["phases"][0]["name"] == "unit"
    assert doc["phases"][0]["duration_s"] > 0
    assert doc["layers"][0]["layer"] == "layer"
    assert doc["summary"]["sample_count"] >= 1


def test_summarize_samples_handles_missing_gpu():
    result = summarize_samples([
        {
            "elapsed_s": 1.0,
            "process_rss_bytes": 100,
            "process_cpu_percent": 25.0,
            "system_cpu_percent": 10.0,
            "system_memory_percent": 50.0,
            "gpu_utilization_percent": None,
            "gpu_memory_used_bytes": None,
            "gpu_memory_total_bytes": None,
        }
    ])
    assert result["peak_process_rss_bytes"] == 100
    assert result["mean_gpu_utilization_percent"] is None
