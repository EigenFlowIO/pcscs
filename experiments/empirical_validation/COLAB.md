# Google Colab execution

The authoritative empirical workflow is now static and repository-native. The human-curated probe sample is committed under `experiments/empirical_validation/static_dataset`; the definitive run must not reacquire, replace, or reselect specimen images.

## Definitive static empirical analysis

Pin the exact Git commit that contains the curated sample and the analysis runner, clone that commit in Colab, install the checkout, verify the frozen image bytes, and run the full empirical analysis.

```python
import subprocess
from google.colab import files

COMMIT = "<VERIFIED_CURATED_STATIC_COMMIT>"
REPO = "/content/pcscs"
OUTPUT = "/content/pcscs_static_empirical_analysis_bundle.zip"

subprocess.run(["git", "clone", "https://github.com/EigenFlowIO/pcscs.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "checkout", COMMIT], check=True)
subprocess.run(["python", "-m", "pip", "install", "-q", f"{REPO}[research,benchmark]"], check=True)

# Verify the exact committed sample. This mode makes no network requests.
subprocess.run([
    "python",
    f"{REPO}/experiments/empirical_validation/cache_static_dataset.py",
    "--verify-only",
], check=True)

# Full 13-layer VGG16/PCSCS analysis, rich hierarchy/tracking outputs,
# bounded dynamic spectral analysis, performance telemetry, and benchmark.
subprocess.run([
    "python",
    f"{REPO}/experiments/empirical_validation/static_empirical_analysis.py",
    "--work-root", "/content/pcscs_static_empirical_analysis_work",
    "--output", OUTPUT,
    "--spectral-steps", "24",
], check=True)

files.download(OUTPUT)
```

The bundle is intended to be returned unchanged for ingestion. It contains the complete scientific layer outputs plus tracking histories, exact filtration-linkage data, dendrogram figures, critical-component image grids, cross-layer pair/sample trajectories, supporting spectral outputs, performance telemetry, benchmark outputs, source hashes, environment metadata, and checksums.

## Static repeatability rerun

After the definitive bundle has been ingested, a second pinned run can verify scientific repeatability. Clone the same commit and run `static_repeatability.py`. It uses the same committed image bytes and does not query GBIF for specimen images.

```python
import subprocess
from google.colab import files

COMMIT = "<SAME_VERIFIED_CURATED_STATIC_COMMIT>"
REPO = "/content/pcscs"
OUTPUT = "/content/pcscs_static_repeatability_bundle.zip"

subprocess.run(["git", "clone", "https://github.com/EigenFlowIO/pcscs.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "checkout", COMMIT], check=True)
subprocess.run(["python", "-m", "pip", "install", "-q", f"{REPO}[research,benchmark]"], check=True)
subprocess.run(["python", f"{REPO}/experiments/empirical_validation/cache_static_dataset.py", "--verify-only"], check=True)
subprocess.run([
    "python",
    f"{REPO}/experiments/empirical_validation/static_repeatability.py",
    "--work-root", "/content/pcscs_static_repeatability_work",
    "--output", OUTPUT,
    "--skip-benchmark",
], check=True)

files.download(OUTPUT)
```
