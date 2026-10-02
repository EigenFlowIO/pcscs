# Changelog
- Commit the exact 250-image CC0/CC BY validation dataset for zero-image-network static repeatability and fix static runner runtime-status initialization.

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
- Made static repeatability repository-native: the committed 250-image dataset is the default input, eliminating notebook upload filenames and dataset-path ambiguity.
