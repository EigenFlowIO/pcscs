# Performance measurement and benchmarking

PCSCS performance has two distinct components that should be measured separately:

1. **End-to-end analysis cost**, which includes model inference, feature handling, similarity construction, PCSCS, and result serialization.
2. **PCSCS-core scaling**, which excludes neural-network feature extraction and measures similarity construction plus PCSCS on fixed representation matrices.

The repository provides separate but compatible instrumentation for both. This avoids attributing model-forward-pass cost to PCSCS itself while still measuring realistic end-to-end workflows.

## End-to-end telemetry

The empirical validation runner records resource telemetry around the exact computation it executes. Instrumentation does not replace or approximate the analytical path.

Recorded fields include:

- total wall-clock time;
- named phase timings;
- VGG16 forward-pass and feature-materialization timing;
- per-layer similarity and PCSCS timing;
- activation shape and flattened representation dimension;
- sample throughput;
- process RAM usage;
- process/system CPU utilization;
- GPU utilization when `nvidia-smi` is available;
- GPU memory reported by both `nvidia-smi` and the PyTorch CUDA allocator;
- Python, platform, PyTorch, CUDA, and device metadata.

Performance output is serialized as JSON and CSV so it can be analyzed independently of console logs.

## Controlled scaling benchmark

`benchmarks/benchmark_pcscs.py` measures the package on deterministic representation matrices. Its default design is:

- sample counts: 25, 50, 100, 200, 400;
- fixed feature dimension: 512;
- one warm-up run per condition;
- five measured runs per condition;
- 250 threshold steps;
- sample tracking disabled for the controlled core benchmark.

Tracking is disabled in the controlled benchmark intentionally. The benchmark is designed to characterize the core threshold/similarity computation without the additional history/tree materialization performed by `SampleTracker`. The end-to-end validation keeps tracking enabled and therefore measures the cost of the full configured workflow separately.

The largest configured deterministic feature matrix is generated once, and smaller sample-count conditions use nested prefixes of that matrix. This holds the representation-generating process fixed as sample count grows. The SHA-256 of each condition matrix is saved with the result, making every timing input auditable.

Run the full benchmark from the repository root:

```bash
python -m pip install -e ".[benchmark]"
python benchmarks/benchmark_pcscs.py
```

For a small smoke run:

```bash
python benchmarks/benchmark_pcscs.py --sample-sizes 10 20
```

Results are written to `benchmark_results/benchmark_results.json` and `benchmark_results/benchmark_runs.csv` by default.

## Interpreting benchmark results

Timing results are hardware- and runtime-specific. Report the actual environment, sample count, representation dimension, threshold count, tracking mode, and package revision alongside timing values.

Do not generalize one measured runtime into universal hardware requirements. In particular, recommendations for larger datasets should be based on measured scaling behavior and observed memory consumption rather than extrapolated hardware tiers.

The sample-count sweep is designed to reveal empirical scaling in the current implementation. It does not by itself prove an asymptotic complexity class. Complexity claims should be grounded in the implementation and algorithm separately from measured wall-clock curves.

## Network-layer performance

When a model exposes multiple layers, model depth alone is not a sufficient explanation of cost. Activation tensor shape and flattened representation dimension can change substantially with depth. The validation telemetry therefore records both layer position and representation size so runtime and memory can be analyzed against each variable.

## Machine-readable schemas

Versioned JSON schemas are provided in `schemas/performance-telemetry.schema.json` and `schemas/benchmark-results.schema.json`. Result bundles also include source hashes, configuration hashes, resolved Git commit information, and SHA-256 checksums so performance measurements can be tied to the exact software and configuration that produced them.
