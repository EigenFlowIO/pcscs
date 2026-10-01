# PCSCS static validation dataset

This directory is the repository location for the accepted frozen image dataset used by `static_repeatability.py`.

The live validation runner creates `/content/pcscs_validation_dataset.zip`. That archive contains the exact accepted image bytes, `sample-manifest.json`, `sample_manifest.csv`, `IMAGE_LICENSES.csv`, `dataset-metadata.json`, and `checksums.sha256`.

After a validation run is accepted, copy the contents of that dataset archive into this directory and commit them. The static repeatability runner never downloads missing images and never falls back to GBIF; a missing or hash-mismatched local image is a hard failure.

The current live validation configuration targets 250 specimens and accepts only media whose exact GBIF multimedia license normalizes to CC0 or CC BY. Third-party image licenses remain separate from the PCSCS software's MIT license.
