# Empirical validation workflow

This directory contains the reproducible PCSCS empirical-validation workflow. The authoritative entry point is `pcscs_colab_validation.py`.

The runner creates a taxonomically stratified arthropod specimen sample from GBIF, freezes the exact sample identity, verifies image hashes and dimensions, extracts VGG16 representations, runs PCSCS across the configured layers, validates result lineage, records performance telemetry, executes the controlled PCSCS scaling benchmark, and emits one compact result bundle.

## Dataset identity

GBIF is a live upstream database, so a future API query is not the definition of a validation dataset. The runner therefore separates discovery from identity:

1. query and archive a candidate pool;
2. deterministically select the configured total sample with adaptive round-robin allocation across family strata;
3. download and validate each selected image;
4. record occurrence identifiers, URLs, dimensions, local filenames, and SHA-256 hashes in `sample-manifest.json`;
5. use that frozen manifest as the exact sample definition for the run.

The configured design targets 250 total specimens across 15 family strata. Family counts are not hard-coded: selection proceeds round-robin across families and automatically redistributes unavailable slots to families with additional valid candidates. The frozen manifest records the realized family counts. To avoid a licensed-media shortage in any one stratum, acquisition may scan multiple GBIF occurrence-search pages while retaining only a bounded deterministic licensed candidate pool per family.

Candidate images are license-filtered before any image download. The current validation configuration accepts only multimedia records whose exact GBIF media license can be normalized to **CC0** or **CC BY**. CC BY-NC, CC BY-SA, unknown/blank licenses, and other restrictive or ambiguous media licenses are excluded from the candidate pool. The selected manifest preserves the original media license, normalized license, creator, rights holder, publisher, media reference, publisher image identifier, GBIF cache URL, and a generated attribution string for each image.

## Analytical path

The authoritative runner uses VGG16 with `IMAGENET1K_V1` weights and the 13 configured convolutional layers. Full flattened activations are written to disk-backed arrays so the representation itself is not changed to satisfy runtime memory constraints. Cosine-similarity matrices are then computed layer by layer and passed to the installed `pcscs` package.

Tracking behavior and threshold resolution come directly from `config.json`.

## Performance telemetry

The same execution records:

- overall and phase-level wall-clock timings;
- image preprocessing and VGG16 forward-pass time;
- per-layer representation dimensionality and materialization time;
- per-layer normalization, cosine-similarity, PCSCS, and serialization time;
- process RAM and CPU utilization;
- GPU utilization and device memory when `nvidia-smi` is available;
- PyTorch CUDA allocator peaks;
- sample throughput and runtime environment metadata.

The runner also invokes `benchmarks/benchmark_pcscs.py` after the scientific validation so the result bundle contains a controlled sample-size scaling benchmark from the same runtime and Git revision.

## Colab

See `COLAB.md` for the single-cell execution command.

## Output artifacts

The default Colab run creates two artifacts:

- `/content/pcscs_validation_bundle.zip` — machine-readable scientific results, telemetry, benchmark results, frozen sample metadata, source hashes, environment metadata, and SHA-256 checksums;
- `/content/pcscs_validation_dataset.zip` — the exact accepted image bytes, frozen manifest, manifest CSV, per-image license/attribution ledger, dataset metadata, and checksums.

Multi-gigabyte activation tensors remain excluded. The dataset ZIP is the intended source for populating the repository's static repeatability dataset after the run is accepted.

## Component scripts

The `scripts/` directory retains smaller workflow components for development, testing, and targeted maintenance. The definitive end-to-end execution is the authoritative runner above; the component scripts are not an alternate validation definition.

## Static repeatability dataset

The repository now contains the accepted **250-image frozen validation dataset** under `static_dataset/`. These are the exact image bytes produced by the definitive licensed acquisition run, not URLs to be resolved later. The directory includes the frozen manifest, per-image license/attribution ledger, dataset metadata, and SHA-256 checksums.

The static repeatability path is intentionally repository-native:

```bash
python experiments/empirical_validation/cache_static_dataset.py --verify-only
python experiments/empirical_validation/static_repeatability.py
```

`static_repeatability.py` reads only `static_dataset/images/`. It performs no GBIF sample query, no specimen-image retrieval, and has no URL fallback. Before feature extraction it verifies the committed bytes against the manifest/checksum set.

`cache_static_dataset.py` remains available as a maintenance/recovery utility for a frozen manifest, but it is **not** part of the normal repeatability path now that the exact image files are committed.

The images are third-party research media and are **not** licensed under the repository's MIT software license. `IMAGE_LICENSES.csv` records the exact media license and attribution metadata for each file. The accepted static set contains only media normalized to CC0 or CC BY by the acquisition workflow.
