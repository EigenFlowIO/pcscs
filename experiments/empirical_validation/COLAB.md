# Google Colab execution

The definitive validation runner is `pcscs_colab_validation.py`. It clones a fresh copy of the public repository at the requested Git ref, records the resolved commit, installs that checkout, runs the complete validation, runs the controlled PCSCS scaling benchmark, and creates one result bundle for downstream analysis.

Use a GPU-backed Colab runtime. Run this single cell:

```python
!rm -rf /content/pcscs && git clone https://github.com/EigenFlowIO/pcscs.git /content/pcscs && cd /content/pcscs && COMMIT=$(git rev-parse HEAD) && python experiments/empirical_validation/pcscs_colab_validation.py --ref "$COMMIT"
```

The runner automatically downloads `pcscs_validation_bundle.zip` when Colab file download is available. The archive is also left at:

```text
/content/pcscs_validation_bundle.zip
```

The bundle includes the frozen sample manifest, source hashes, scientific result arrays, layer metrics, full performance telemetry, the controlled scaling benchmark, environment metadata, and SHA-256 checksums.

For a quick infrastructure-only check of the benchmark path, the validation runner accepts `--benchmark-sample-sizes`, but a reduced size list is not a substitute for the configured definitive execution.
