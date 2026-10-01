from __future__ import annotations
import math
from common import CONFIG_PATH, MANIFEST_PATH, RESULTS_DIR, load_json, config_hash, sha256_file

def main():
    cfg=load_json(CONFIG_PATH); summary=load_json(RESULTS_DIR / "summary.json")
    errors=[]
    if summary.get("config_sha256") != config_hash(): errors.append("config hash mismatch")
    if summary.get("sample_manifest_sha256") != sha256_file(MANIFEST_PATH): errors.append("sample manifest hash mismatch")
    if summary.get("n_samples") != cfg["target_total"]: errors.append("sample count mismatch")
    expected=set(cfg["model"]["layers"]); got=set(summary.get("layers",{}))
    if got != expected: errors.append(f"layer mismatch expected={sorted(expected)} got={sorted(got)}")
    for layer, row in summary.get("layers",{}).items():
        for key in ("convergence_threshold","critical_threshold","critical_rate"):
            v=row.get(key)
            if not isinstance(v,(int,float)) or not math.isfinite(v): errors.append(f"{layer}: invalid {key}")
        npz=RESULTS_DIR / f"{layer.replace('.', '_')}.npz"
        if not npz.exists(): errors.append(f"{layer}: missing {npz.name}")
    if errors: raise SystemExit("Result validation FAILED:\n- " + "\n- ".join(errors))
    print(f"Result validation PASSED: {len(got)} layers, manifest/config lineage verified")

if __name__ == "__main__": main()
