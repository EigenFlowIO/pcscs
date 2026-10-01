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

A reproducible reference run was completed on 2026-10-01 from repository commit
`fb2867b197cbfd5b67beb5a7855b64eb8a873c08`.

Runtime environment:

- Tesla T4 GPU;
- 14.56 GiB reported GPU memory;
- 12.67 GiB host RAM;
- 2 logical CPUs;
- Python 3.13.15;
- PyTorch 2.11.0 with CUDA 13.0.

The instrumented end-to-end workflow analyzed 180 specimens across 13 VGG16 convolutional layers.
Telemetry covered 499.27 seconds (8.32 minutes). Peak process RSS was 7.23 GiB. Mean GPU
utilization was 4.63%, peak GPU utilization was 36%, and peak GPU memory used was 795.0 MiB.

The low average GPU utilization is specific to this implementation and runtime. The full workflow
contains substantial disk-backed feature handling, normalization, similarity construction, and
CPU-side PCSCS analysis, so it is not expected to saturate the GPU continuously.

### Controlled sample-count benchmark

The controlled benchmark used 512-dimensional deterministic feature matrices, 250 threshold steps,
tracking disabled, one warm-up, and five measured repetitions per condition.

| Samples | Median total time | Median PCSCS time | Peak process RSS |
| ---: | ---: | ---: | ---: |
| 25 | 0.0366 s | 0.0363 s | 559.2 MiB |
| 50 | 0.1330 s | 0.1326 s | 559.4 MiB |
| 100 | 0.5854 s | 0.5776 s | 561.6 MiB |
| 200 | 1.9620 s | 1.9585 s | 563.0 MiB |
| 400 | 9.1972 s | 9.1928 s | 572.9 MiB |

A log-log fit over this measured range gives an empirical slope of approximately 1.98 for both
total and PCSCS-only median runtime. This is consistent with approximately quadratic scaling over
the tested range. It should not be interpreted as an empirical proof of asymptotic complexity.

### Layer representation size and cost

Across the 13 VGG16 layers, flattened activation dimensionality ranged from 3,211,264 values in the
earliest analyzed layers to 100,352 in the deepest analyzed layers. Flattened dimension was strongly
associated with layer-associated runtime (Pearson `r = 0.969`; Spearman `rho = 0.981`) and especially
with cosine-similarity computation time (Pearson `r = 0.997`).

These measurements support a practical interpretation: for this workflow, representation size is a
more direct predictor of computational cost than layer depth by itself.

### Scope

These values describe the measured Tesla T4 Colab environment and the exact benchmark configuration
above. They are not universal hardware requirements. Broader architecture comparisons require separate
measurements rather than extrapolation from this reference run.
