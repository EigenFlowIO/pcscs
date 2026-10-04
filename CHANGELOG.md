# Changelog

## Unreleased

- Replace the prior static validation sample with the authoritative 250-image human-curated probe set; exact duplicate image bytes and visually unsuitable label/document/non-specimen images are excluded before analysis.
- Add `static_empirical_analysis.py`, the repository-native definitive runner for the curated sample. It preserves full PCSCS tracking and merge histories, filtration-faithful hierarchy outputs, critical-component image grids, cross-layer pair/sample trajectories, supporting spectral outputs, and performance evidence.
- Add exact threshold-filtration linkage utilities so dendrogram cuts correspond to PCSCS connected components rather than an independent average-linkage clustering.
- Update Colab guidance so definitive analysis clones a pinned repository commit and runs only against the committed static sample; live sample acquisition is no longer the authoritative manuscript workflow.

## 0.2.0 - 2026-10-01

- Commit the exact 250-image CC0/CC BY validation dataset for zero-image-network static repeatability and fix static runner runtime-status initialization.
- Updated public terminology to describe the critical threshold as a descriptive statistic, including boundary-limited cases.
- Replaced the superseded 180-sample performance reference with the definitive n=250 Tesla T4 validation and scaling measurements.
- Revalidated repository-native static repeatability documentation against the frozen 250-image CC0/CC BY dataset.


## 0.1.0

- Initial public release of the `pcscs` Python package.
- Added progressive cosine-similarity threshold analysis, connected-component tracking, critical-threshold estimation, visualization, PyTorch feature extraction, and graph-spectral utilities.
- Added standard Python packaging, automated tests, CI, examples, citation metadata, and an end-to-end deterministic empirical-validation workflow.
- Corrected the linear-interpolation fallback to respect NumPy's increasing-x requirement.
- Corrected spectral connected-component reporting so partial eigendecomposition does not cap the graph component count at `k`.
- Retained historical public API field names where required for compatibility; documentation defines their current semantics.
- Added reproducible performance telemetry, a deterministic sample-size scaling benchmark, Colab execution guidance, and machine-readable performance schemas.
- Reworked GBIF sample freezing so family counts are adaptive rather than hard quotas: selection now proceeds deterministically round-robin across configured family strata until the total sample target is reached.
- Switched empirical image acquisition to the GBIF occurrence-image cache using a single shared HTTP session; publisher media identifiers remain frozen as provenance and sample-selection diagnostics record cache failures.
- Added a measured Tesla T4 reference performance profile and fixed benchmark CSV export so condition sample counts cannot be overwritten by telemetry sample counts.

- Added a frozen static validation-sample definition plus one-time image materialization and hash verification tooling.
- Added `static_repeatability.py`, which reruns the scientific validation from committed image bytes without GBIF sample queries or image retrieval.
- Restricted new empirical-validation candidate media to explicitly redistributable CC0 and CC BY licenses, with per-image license/attribution metadata preserved in the frozen manifest.
- Increased the configured validation sample target from 180 to 250 specimens and expanded licensed candidate discovery through paginated GBIF scans.
- Added a separate `pcscs_validation_dataset.zip` artifact containing the exact accepted image bytes, image-license ledger, frozen manifest, and checksums for later static repeatability.
- Expanded licensed-image discovery to scan up to 900 GBIF occurrence records per family while retaining a bounded deterministic pool of up to 120 license-compatible candidates per family.
- Added explicit per-image attribution text and stronger dataset-package tests for the 250-image static validation workflow.
- Removed the incomplete 180-image static-dataset placeholder so only an accepted frozen dataset is installed into `static_dataset/`.
- Made static repeatability repository-native: the committed 250-image dataset is the default input, eliminating upload-filename and dataset-path ambiguity.
