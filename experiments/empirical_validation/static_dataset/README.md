# Authoritative static empirical probe sample

This directory contains the exact **250 image files** in the human-curated probe sample used by the repository-native empirical validation. The sample is an analytical instrument for interrogating VGG16 internal representation geometry; it is not a biological-classification benchmark.

## Curation gate

Candidate preserved-specimen images were drawn under the configured family-stratified acquisition design and exact-media CC0/CC-BY license gate. A human reviewer then inspected candidates one at a time and rejected label/document-dominated, non-specimen, badly cropped, severely obscured, or otherwise unsuitable images. Exact duplicate image bytes were rejected by SHA-256 before human review. Acquisition continued until 250 approved images remained.

## Reproducibility

`sample-manifest.json` is the normalized manifest consumed by the empirical runner. `curation-manifest.json`, `curation-config.json`, `human_rejections.csv`, and `automatic_skips.csv` preserve the curation provenance. `images/` contains the exact frozen bytes. `checksums.sha256` verifies the complete static dataset payload.

The definitive run must clone a pinned repository commit and execute `static_empirical_analysis.py`; it must not reacquire specimen images.

The image files are third-party media and are not covered by the repository MIT software license.
