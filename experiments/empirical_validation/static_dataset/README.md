# PCSCS static validation dataset

This directory contains the exact **250 image files** used in the definitive license-filtered empirical validation. The image bytes live under `static_dataset/images/`. The files are committed so the static repeatability workflow does not depend on GBIF or any external image server.

`sample-manifest.json` records the frozen sample and exact image SHA-256 hashes. `IMAGE_LICENSES.csv` records the per-image source license and attribution metadata. `checksums.sha256` covers the dataset artifact files emitted by the definitive validation run.

The definitive acquisition policy admitted only exact multimedia records whose licenses normalized to **CC0** or **CC BY**. Third-party image files retain those licenses and are **not licensed under the PCSCS MIT software license**.

Run static repeatability from the repository root with:

```bash
python experiments/empirical_validation/static_repeatability.py
```

The static runner validates the local files before analysis and never falls back to remote image URLs.

Repository-local image bytes live in `static_dataset/images/`; the static runner reads those files directly.
