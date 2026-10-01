from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import CONFIG_PATH, CANDIDATE_POOL_PATH, load_json
from pcscs_colab_validation import build_candidate_pool


def main():
    cfg = load_json(CONFIG_PATH)
    path = build_candidate_pool(cfg, CONFIG_PATH, CANDIDATE_POOL_PATH.parent)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
