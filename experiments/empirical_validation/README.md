# Empirical validation workflow

This directory contains the reproducible PCSCS empirical validation workflow. It constructs a balanced arthropod specimen dataset from GBIF, freezes the exact selected sample, validates the sample cryptographically, extracts VGG16 representations, runs PCSCS, and validates the resulting outputs.

## Reproducibility model

GBIF is a live upstream database, so a future query is not considered the definition of a validation dataset. The workflow separates dataset discovery from dataset identity:

1. `build_candidate_pool.py` queries GBIF and archives the candidate pool returned at that time.
2. `freeze_sample_manifest.py` deterministically selects the configured sample and writes `data/sample-manifest.json`.
3. The manifest records occurrence identifiers, media URLs, dimensions, local filenames, and SHA-256 hashes.
4. `materialize_sample.py` reconstructs the exact frozen sample from the manifest.
5. `validate_sample.py` verifies sample size, balance, uniqueness, dimensions, configuration lineage, and image hashes.
6. `run_validation.py` performs VGG16 feature extraction and PCSCS analysis.
7. `validate_results.py` checks the generated result set and its lineage.

Once `sample-manifest.json` is frozen, it is the canonical definition of the dataset used for that validation run.

## GitHub Actions

Two manually triggered workflows are provided:

- `freeze-validation-sample` performs the live GBIF query, freezes and validates the configured sample, and uploads the candidate pool, manifest, and exact image archive as a workflow artifact.
- `run-validation` requires an approved committed `sample-manifest.json`, rematerializes and hash-validates the exact sample, runs VGG16 + PCSCS, validates the outputs, freezes environment metadata, and uploads the results.

Recommended sequence:

1. Run `freeze-validation-sample`.
2. Download and inspect the resulting artifact.
3. Commit the approved `candidate-pool.json` and `sample-manifest.json` to `data/`.
4. Run `run-validation`.
5. Commit the validated summaries, tables, and figures that should remain part of the repository record; keep bulky transient arrays and source images outside ordinary Git history.

## Local execution

The same scripts can be run locally from the repository root after installing the research dependencies.
