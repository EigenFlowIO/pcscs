from __future__ import annotations
import json, os, platform, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights
from pcscs import PCSCS
from pcscs.extraction import extract_features_from_model
from common import CONFIG_PATH, MANIFEST_PATH, IMAGE_DIR, RESULTS_DIR, load_json, dump_json, config_hash, sha256_file


def set_determinism(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main():
    cfg = load_json(CONFIG_PATH); manifest = load_json(MANIFEST_PATH)
    set_determinism(0)
    weights = VGG16_Weights.IMAGENET1K_V1
    if cfg["model"]["torchvision_weights"] != "IMAGENET1K_V1":
        raise RuntimeError("config requests unimplemented VGG16 weights")
    model = vgg16(weights=weights)
    preprocess = weights.transforms()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = sorted(manifest["samples"], key=lambda x: x["sample_index"])
    images = [Image.open(IMAGE_DIR / r["filename"]).convert("RGB") for r in rows]
    labels = [{k:r[k] for k in ("family","imagenet_class","imagenet_index","occurrence_key")} for r in rows]
    layers = cfg["model"]["layers"]
    layer_features = extract_features_from_model(images, model, preprocess, layers, device=device)
    analyzer = PCSCS(enable_tracking=bool(cfg["pcscs"]["enable_tracking"]))
    results = analyzer.analyze_layers(layer_features, sample_labels=labels, n_steps=int(cfg["pcscs"]["n_steps"]))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_id": cfg["experiment_id"],
        "config_sha256": config_hash(),
        "sample_manifest_sha256": sha256_file(MANIFEST_PATH),
        "n_samples": len(rows),
        "device": device,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision_weights": cfg["model"]["torchvision_weights"],
        "layers": {}
    }
    for name, r in results.items():
        summary["layers"][name] = {
            "convergence_threshold": float(r.convergence_threshold),
            "critical_threshold": float(r.critical_threshold),
            "critical_rate": float(r.critical_rate),
            "fit_successful": bool(r.fit_successful)
        }
        np.savez_compressed(
            RESULTS_DIR / f"{name.replace('.', '_')}.npz",
            thresholds=r.thresholds,
            num_components=r.num_classes,
            smooth_thresholds=r.smooth_thresholds,
            smooth_components=r.smooth_classes,
            derivative=r.derivative
        )
    dump_json(RESULTS_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
