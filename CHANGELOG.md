# Changelog

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
- Increased the configured validation sample target from 180 to 250 specimens and expanded candidate discovery to 300 records per family stratum.
- Added a separate `pcscs_validation_dataset.zip` artifact containing the exact accepted image bytes, image-license ledger, frozen manifest, and checksums for later static repeatability.
- Removed the incomplete 180-image static-dataset placeholder so only an accepted frozen dataset is installed into `static_dataset/`.
