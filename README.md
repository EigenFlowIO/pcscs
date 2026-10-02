# PCSCS

Progressive Cosine Similarity Classification Sifting (PCSCS) is a post-hoc neural-network analysis toolkit for studying how samples are organized in learned representation spaces.

PCSCS treats a set of neural representations as a similarity graph. At a chosen cosine-similarity threshold, samples are connected when their pairwise similarity exceeds that threshold. By sweeping the threshold from strict to permissive values, PCSCS tracks how connected components merge, reports descriptive transition summaries, and exposes sample-level and layer-level organization that is difficult to see from predictions alone.

The package is designed for empirical analysis, quantitative comparison, statistical hypothesis workflows, and neural-network behavior interpretation. It is model-agnostic at the analysis stage: any representation matrix can be analyzed once feature vectors have been extracted.

## Installation

From a clone of the repository:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

For the included image-model validation workflow:

```bash
python -m pip install -e ".[research]"
```

## Core idea

Let each sample be represented by a feature vector \(h_i\). PCSCS begins with the cosine-similarity matrix

\[
S(i,j) = \frac{h_i \cdot h_j}{\|h_i\|\,\|h_j\|}.
\]

For a threshold \(\theta\), PCSCS forms a graph whose vertices are samples and whose edges satisfy

\[
S(i,j) > \theta.
\]

As \(\theta\) decreases, edges are added and connected components merge. PCSCS records this evolution across a dense threshold sequence.

The principal outputs are:

- the convergence threshold, where all samples become connected;
- the connected-component count across thresholds;
- a smoothed component-count curve;
- a critical threshold corresponding to the maximum derivative of the smoothed curve with respect to increasing similarity threshold;
- merge events and sample trajectories when tracking is enabled;
- optional graph-spectral summaries and visualizations.

Historical API fields such as `num_classes` are retained for compatibility. In PCSCS they represent connected-component counts, not supervised class predictions.

## Minimal analysis

```python
import numpy as np
from pcscs import PCSCS, compute_similarity_matrix

features = np.array([
    [1.00, 0.00],
    [0.96, 0.08],
    [0.08, 0.96],
    [0.00, 1.00],
])

similarity = compute_similarity_matrix(features)

analyzer = PCSCS(enable_tracking=True)
result = analyzer.analyze_layer(
    similarity,
    layer_name="embedding",
    sample_labels=["a", "b", "c", "d"],
    n_steps=500,
)

print("convergence:", result.convergence_threshold)
print("critical threshold:", result.critical_threshold)
print("critical rate:", result.critical_rate)
```

A runnable example is provided in `examples/synthetic_demo.py`.

## Public API

The top-level package exports:

```python
from pcscs import (
    PCSCS,
    fit_sigmoid,
    find_critical_threshold,
    sigmoid_function,
    compute_similarity_matrix,
)
```

### `PCSCS`

`PCSCS(enable_tracking=True)` is the main analyzer.

Important methods:

- `find_convergence_threshold(similarity_matrix, start_threshold=0.999, step_size=0.02)`
- `progressive_sift(similarity_matrix, convergence_threshold, n_steps=1000, sample_labels=None)`
- `analyze_layer(similarity_matrix, layer_name="Layer", sample_labels=None, n_steps=1000)`
- `analyze_layers(layer_features, sample_labels=None, n_steps=1000)`

`analyze_layer` returns a `PCSCSResults` object with the following primary fields:

- `layer_name`
- `convergence_threshold`
- `critical_threshold`
- `critical_rate`
- `thresholds`
- `num_classes`
- `smooth_thresholds`
- `smooth_classes`
- `sigmoid_params`
- `fit_successful`
- `derivative`
- `tracking_data`

### Similarity construction

`compute_similarity_matrix(features)` accepts a two-dimensional feature array and returns the pairwise cosine-similarity matrix used by PCSCS.

### Curve fitting and transition analysis

`fit_sigmoid(thresholds, num_classes)` fits the threshold-versus-component-count sequence with a sigmoid when possible and returns smooth arrays, fitted parameters, and a success flag.

`find_critical_threshold(smooth_thresholds, smooth_classes, sigmoid_params=None)` computes the threshold at which the smoothed component-count curve has its maximum derivative with respect to increasing similarity threshold.

The critical threshold is a descriptive structural statistic. It should be interpreted in relation to the representation, sample set, and threshold construction used in an analysis. If the maximum derivative occurs at the boundary of the analyzed threshold interval, the reported value is boundary-limited rather than evidence of an interior transition point.

## Feature extraction from PyTorch models

`pcscs.extraction` provides feature extraction through PyTorch forward hooks.

```python
from pcscs.extraction import extract_features_from_model

layer_features = extract_features_from_model(
    samples=samples,
    model=model,
    preprocess_fn=preprocess,
    layer_names=["features.0", "features.2", "features.5"],
    device="cpu",
)

results = PCSCS(enable_tracking=True).analyze_layers(
    layer_features,
    sample_labels=labels,
    n_steps=500,
)
```

If `layer_names` is omitted, the extractor attempts to identify convolutional layers automatically.

Each returned layer entry contains:

- `features`
- `similarity_matrix`
- `feature_shape`
- `n_samples`

The current extractor processes samples individually. Users with large datasets may prefer to batch feature extraction externally and pass the resulting feature matrices directly to PCSCS.

## Sample tracking and merge analysis

When `enable_tracking=True`, PCSCS records how components evolve across the threshold sweep.

`pcscs.tracking` provides:

- `SampleTracker`
- `analyze_sample_trajectories`
- `find_most_similar_samples`

Tracking data can be used to study questions such as:

- which samples merge earliest;
- which samples remain isolated longest;
- which groups are stable across threshold ranges;
- which sample pairs exhibit the strongest representational similarity;
- how component structure changes across network depth.

These outputs can support empirical hypothesis tests by supplying measurable structural quantities that can be compared across conditions, models, layers, datasets, or controlled perturbations. PCSCS does not prescribe a specific inferential test; users may apply statistical procedures appropriate to their experimental design.

## Visualization

`pcscs.visualization` includes utilities for inspecting threshold dynamics and component structure:

- `plot_layer_analysis`
- `plot_merge_events`
- `plot_dendrogram`
- `plot_combined_layers`
- `get_similarity_class_partitions`
- `print_class_summary`

Example:

```python
from pcscs.visualization import plot_layer_analysis, plot_dendrogram

plot_layer_analysis(result)
plot_dendrogram(result, test_threshold=result.critical_threshold)
```

The term “similarity class” appears in some compatibility functions. It refers to a connected component of the threshold graph.

## Spectral analysis

`pcscs.spectral` provides optional graph-spectral tools for examining the same similarity structure through Laplacian eigenvalues.

The `SpectralAnalyzer` supports:

- graph-Laplacian construction;
- static spectral analysis at one threshold;
- dynamic spectral analysis across thresholds;
- connected-component estimation from the Laplacian spectrum;
- algebraic connectivity and spectral-gap analysis;
- comparison of spectral transitions with PCSCS critical thresholds.

Visualization helpers include:

- `plot_eigenvalue_spectrum`
- `plot_spectral_flow`
- `plot_spectral_properties`

Spectral analysis is complementary to the core PCSCS threshold-component analysis and can be used when a graph-theoretic view of representational organization is useful.

## Empirical and statistical workflows

PCSCS is intended to support controlled empirical analysis of learned representations. Typical uses include:

- comparing representational organization across layers;
- comparing architectures trained on the same task;
- measuring how augmentation, perturbation, domain shift, or fine-tuning changes component structure;
- testing whether designed probe groups merge in the expected order;
- quantifying stability of sample relationships across network depth;
- identifying unusually isolated or unusually similar samples;
- comparing critical thresholds or trajectory-derived summaries across experimental conditions.

A rigorous workflow should define the sample set, representation source, preprocessing, threshold resolution, model version, and statistical comparison procedure before interpreting differences.

## Reproducible validation workflow

`experiments/empirical_validation/` contains a deterministic end-to-end validation pipeline built around preserved museum specimens from GBIF and VGG16 representations.

The workflow separates live data discovery from dataset identity. The current design targets 250 specimens across 15 arthropod family strata. Candidate media are eligible only when the exact GBIF multimedia record carries a redistribution-friendly license normalized to CC0 or CC BY; unknown, non-commercial, share-alike, and other ambiguous/restrictive media licenses are rejected before image selection. A deterministic sample is frozen into a manifest, and every accepted image is recorded with provenance metadata, license/attribution metadata, dimensions, and a SHA-256 hash.

The authoritative Colab runner, `experiments/empirical_validation/pcscs_colab_validation.py`, performs the complete live acquisition-and-validation path and emits two artifacts: a scientific result bundle and a frozen dataset bundle containing the exact accepted image bytes. It records the resolved Git commit, freezes and validates the specimen sample, extracts VGG16 representations, runs PCSCS, validates the outputs, captures end-to-end performance telemetry, and runs the controlled scaling benchmark.

For static repeatability, the accepted dataset bundle is copied into `experiments/empirical_validation/static_dataset/` and committed with its manifest, checksums, and image-license ledger. `static_repeatability.py` then reruns the same scientific-analysis functions from those repository-local images. It never performs sample selection, never retrieves missing specimen images, and treats missing or hash-mismatched local files as a hard failure.

See `experiments/empirical_validation/README.md` and `experiments/empirical_validation/COLAB.md` for execution details.

## Performance measurement

PCSCS separates realistic end-to-end telemetry from controlled core-algorithm benchmarking. The validation runner measures the exact configured workflow, including VGG16 inference, feature materialization, per-layer similarity construction, PCSCS analysis, CPU/RAM metrics, and GPU utilization/memory when available.

`benchmarks/benchmark_pcscs.py` separately measures sample-size scaling on deterministic fixed representation matrices. The default benchmark uses sample sizes 25, 50, 100, 200, and 400, with one warm-up and five measured runs per condition.

See `docs/performance.md` for the measurement protocol, output format, and interpretation constraints.

## Practical interpretation

PCSCS measures organization in representation space, not causal mechanism by itself. Results are conditional on the selected representation, similarity metric, sample population, preprocessing, and threshold discretization.

The convergence threshold is currently found by a decrementing grid search. The default implementation scans from `0.999` downward in steps of `0.02`, so the returned value is an approximation at that resolution.

Cosine similarity is appropriate when angular similarity is meaningful for the representation being studied. Other representation geometries may require a different similarity or distance construction before applying an analogous threshold-graph analysis.

## Repository structure

```text
src/pcscs/                         package source
examples/                          runnable examples
experiments/empirical_validation/ deterministic validation workflow
benchmarks/                         controlled performance benchmark
docs/                              package documentation
tests/                             automated tests
.github/workflows/                 CI and validation workflows
```

## Testing

Run the test suite with:

```bash
pytest
```

The repository also includes GitHub Actions for package tests and the empirical validation workflow.

## Citation

Software citation metadata is provided in `CITATION.cff`.

## License

PCSCS is released under the MIT License. Use, modification, and redistribution are permitted provided the copyright and permission notice are retained. See `LICENSE`.
