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
- Added multi-URL media fallback, browser-compatible request headers, retries, parallel per-family acquisition, and machine-readable sample-selection diagnostics to reduce provider-specific image-access failures.
