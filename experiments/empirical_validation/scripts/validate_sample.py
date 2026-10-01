from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import CONFIG_PATH, MANIFEST_PATH, IMAGE_DIR, load_json
from pcscs_colab_validation import validate_sample


def main():
    cfg = load_json(CONFIG_PATH)
    validate_sample(cfg, CONFIG_PATH, MANIFEST_PATH, IMAGE_DIR)


if __name__ == "__main__":
    main()
