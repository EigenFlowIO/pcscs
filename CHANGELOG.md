# Changelog

## 0.1.0

- Initial public release of the `pcscs` Python package.
- Added progressive cosine-similarity threshold analysis, connected-component tracking, critical-threshold estimation, visualization, PyTorch feature extraction, and graph-spectral utilities.
- Added standard Python packaging, automated tests, CI, examples, citation metadata, and an end-to-end deterministic empirical-validation workflow.
- Corrected the linear-interpolation fallback to respect NumPy's increasing-x requirement.
- Corrected spectral connected-component reporting so partial eigendecomposition does not cap the graph component count at `k`.
- Retained historical public API field names where required for compatibility; documentation defines their current semantics.
