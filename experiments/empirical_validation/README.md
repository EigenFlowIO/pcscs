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

The configured design targets 180 total specimens across 15 family strata. Family counts are not hard-coded: selection proceeds round-robin across families and automatically redistributes unavailable slots to families with additional valid candidates. The frozen manifest records the realized family counts.

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

## Output bundle

The default Colab output is `/content/pcscs_validation_bundle.zip`. It contains machine-readable scientific results, telemetry, benchmark results, frozen sample metadata, source hashes, environment metadata, and SHA-256 checksums. Raw source images and multi-gigabyte activation tensors are excluded from the bundle; the frozen manifest preserves the information required to reconstruct and verify the exact image sample.

## Component scripts

The `scripts/` directory retains smaller workflow components for development, testing, and targeted maintenance. The definitive end-to-end execution is the authoritative runner above; the component scripts are not an alternate validation definition.

## Static repeatability dataset

The live validation runner remains the acquisition-and-validation workflow. A separate static repeatability path is provided so later runs can use the exact accepted image bytes without querying GBIF or contacting image servers.

The frozen definition lives in `static_dataset/sample-manifest.json`. The exact 180 images belong in `static_dataset/images/`; each filename, SHA-256 hash, and image dimensions are fixed by that manifest.

To materialize the image set once from the already frozen manifest:

```bash
python experiments/empirical_validation/cache_static_dataset.py
```

The cache utility does **not** perform sample selection. It requests only the GBIF cache URLs already recorded in the definitive manifest and rejects any downloaded file whose SHA-256 or dimensions differ from the recorded sample. It also writes `checksums.sha256`, `dataset-metadata.json`, and `IMAGE_LICENSES.csv`.

After the images are present, an offline image-sample verification can be run with:

```bash
python experiments/empirical_validation/cache_static_dataset.py --verify-only
```

The repeatability analysis is then run with:

```bash
python experiments/empirical_validation/static_repeatability.py
```

`static_repeatability.py` performs no GBIF sample query and no image download. It verifies the committed static dataset first, then invokes the same VGG16 feature-extraction and PCSCS scientific-analysis functions used by `pcscs_colab_validation.py`.

The static image set is intentionally separate from the live acquisition logic: the live runner can construct a new empirical sample when needed, while the static runner reproduces the frozen reference sample byte-for-byte.

Before distributing the cached image files publicly, review `IMAGE_LICENSES.csv` and the source media rights. The cache utility records available GBIF/media rights metadata but does not make a legal determination about redistribution.
