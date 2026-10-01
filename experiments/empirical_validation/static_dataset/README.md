# PCSCS static validation sample v1

This directory defines the frozen image sample used for static PCSCS repeatability.

`sample-manifest.json` is copied from the successful definitive validation run based on source commit `fb2867b197cbfd5b67beb5a7855b64eb8a873c08`. It defines 180 accepted specimens and records each local filename, GBIF occurrence key, source media identifier/cache URL, dimensions, and SHA-256 hash.

The `images/` directory is designed to contain those exact 180 image files. Images are not considered part of the frozen dataset unless every file passes the hashes and dimensions in the manifest.

Run:

```bash
python experiments/empirical_validation/cache_static_dataset.py
```

to materialize the exact image bytes from the already frozen GBIF cache URLs. No new sample selection occurs.

After materialization, run:

```bash
python experiments/empirical_validation/cache_static_dataset.py --verify-only
```

to perform a network-free integrity check.

`reference_results/` contains the scientific summary and layer metrics from the definitive run for downstream comparison. The repeatability runner itself uses the static images and the current package implementation; it does not retrieve specimen images from external servers.

`IMAGE_LICENSES.csv` is generated during materialization from available source metadata. Review the recorded licenses and rights before redistributing the image bytes publicly.
