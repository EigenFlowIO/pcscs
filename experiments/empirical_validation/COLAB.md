# Google Colab execution

Two Colab workflows are intentionally distinct.

## Live acquisition / definitive empirical validation

`pcscs_colab_validation.py` queries GBIF, constructs a licensed sample, retains the accepted image bytes, runs the scientific validation and controlled benchmark, and emits both a result bundle and a dataset bundle. Pin the exact Git commit.

```python
import subprocess
from google.colab import files

COMMIT = "<VERIFIED_GIT_COMMIT>"
RUNNER = "/content/pcscs_colab_validation.py"

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

files.download("/content/pcscs_validation_bundle.zip")
files.download("/content/pcscs_validation_dataset.zip")
```

## Static repeatability from the repository

After the frozen dataset has been committed, do **not** upload a dataset ZIP and do **not** contact GBIF for specimen images. Clone the repository at the exact commit and run the static validator against the image files already in the repository.

```python
import subprocess
from google.colab import files

COMMIT = "<VERIFIED_STATIC_DATASET_COMMIT>"
REPO = "/content/pcscs"
OUTPUT = "/content/pcscs_static_repeatability_bundle.zip"

subprocess.run(["git", "clone", "https://github.com/EigenFlowIO/pcscs.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "checkout", COMMIT], check=True)
subprocess.run(["python", "-m", "pip", "install", "-q", f"{REPO}[research,benchmark]"], check=True)

subprocess.run([
    "python",
    f"{REPO}/experiments/empirical_validation/cache_static_dataset.py",
    "--verify-only",
], check=True)

subprocess.run([
    "python",
    f"{REPO}/experiments/empirical_validation/static_repeatability.py",
    "--work-root", "/content/pcscs_static_repeatability_work",
    "--output", OUTPUT,
    "--skip-benchmark",
], check=True)

files.download(OUTPUT)
```

The static runner's default dataset path is the committed `experiments/empirical_validation/static_dataset` directory. A filename/path mismatch therefore cannot arise from a notebook upload name.
