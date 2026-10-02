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

## Reference measurement: Colab Tesla T4

A definitive licensed-image validation run was completed on 2026-10-01 from source commit
`d761a31e4a821f8b193e4eb1860e3e2f2d05dc5d`. The frozen 250-image dataset was subsequently committed for repository-native static repeatability.

Runtime environment:

- Tesla T4 GPU;
- 15 GiB reported GPU memory;
- Python 3.13.15;
- PyTorch 2.11.0 with CUDA 13.0;
- VGG16 `IMAGENET1K_V1`;
- 250 specimens across 15 family strata;
- 13 VGG16 convolutional layers.

The instrumented end-to-end workflow covered **800.49 s (13.34 min)**. Major phases included 29.60 s for candidate-pool querying, 143.33 s for sample freezing/download, 71.58 s for feature extraction/materialization, 546.53 s for scientific validation, and 79.20 s for the controlled scaling benchmark. Phase timers may be nested and should not be summed indiscriminately.

Peak process RSS was **7.38 GiB**. Mean GPU utilization was **4.98%**, peak GPU utilization was **36%**, and peak GPU memory used was approximately **795 MiB**. These values describe this implementation/runtime, not a universal hardware requirement.

### Controlled sample-count benchmark

The controlled benchmark used deterministic 512-dimensional feature matrices, 250 threshold steps, tracking disabled, one warm-up, and five measured repetitions.

| Samples | Median total time | Median PCSCS time |
| ---: | ---: | ---: |
| 25 | 0.0354 s | 0.0350 s |
| 50 | 0.1404 s | 0.1399 s |
| 100 | 0.5491 s | 0.5481 s |
| 200 | 2.0021 s | 2.0007 s |
| 400 | 9.3359 s | 9.3315 s |

A log-log fit of median total time against sample count gives an empirical slope of approximately **1.99** over n=25-400. This is consistent with approximately quadratic timing over the measured range; it is not a proof of asymptotic complexity.

### Layer representation size and cost

Flattened activation dimensionality ranged from 3,211,264 values in the earliest analyzed layers to 100,352 in the deepest analyzed layers. Across the 13 layers, feature dimension was strongly associated with layer-associated runtime (Pearson `r = 0.994`; Spearman `rho = 0.981`) and with cosine-similarity computation time (Pearson `r = 0.997`).

This supports a practical interpretation: in this workflow, representation size is a more direct predictor of layer cost than network depth alone.

### Static repeatability timing

A separate repository-native static repeatability execution using the committed image bytes completed in **549.37 s**. It omitted live sample acquisition and the controlled benchmark. Scientific result equality, rather than identical timing, is the repeatability criterion.

### Scope

These measurements support claims about the recorded Tesla T4 environment and the exact configurations above. Broader architecture or hardware claims require separate measurements.
