# Google Colab execution

The definitive validation runner is `pcscs_colab_validation.py`. It clones a fresh copy of the public repository at the requested Git ref, records the resolved commit, installs that checkout, runs the complete validation, runs the controlled PCSCS scaling benchmark, and creates one result bundle for downstream analysis.

Use a GPU-backed Colab runtime. Run this single **Python** cell:

```python
import subprocess
from google.colab import files

COMMIT = "<VERIFIED_GIT_COMMIT>"
RUNNER = "/content/pcscs_colab_validation.py"
BUNDLE = "/content/pcscs_validation_bundle.zip"
DATASET = "/content/pcscs_validation_dataset.zip"

subprocess.run([
    "wget", "-q",
    f"https://raw.githubusercontent.com/EigenFlowIO/pcscs/{COMMIT}/experiments/empirical_validation/pcscs_colab_validation.py",
    "-O", RUNNER,
], check=True)

subprocess.run([
    "python", RUNNER,
    "--ref", COMMIT,
    "--no-download",
], check=True)

files.download(BUNDLE)
files.download(DATASET)
```

The explicit commit pin is required for the definitive run. The runner itself clones the repository at that exact commit and records the resolved commit in the output bundle.

The results bundle contains the frozen sample manifest, source hashes, scientific result arrays, layer metrics, full performance telemetry, the controlled scaling benchmark, environment metadata, and SHA-256 checksums. The dataset bundle contains the exact accepted image bytes plus per-image license and attribution metadata so the static repeatability dataset can be committed without depending on future remote-image availability.

For a quick infrastructure-only check of the benchmark path, the validation runner accepts `--benchmark-sample-sizes`, but a reduced size list is not a substitute for the configured definitive execution.
