import numpy as np
from pcscs import PCSCS, compute_similarity_matrix

features = np.array([
    [1.0, 0.0],
    [0.98, 0.08],
    [0.0, 1.0],
    [0.08, 0.98],
    [0.70, 0.70],
], dtype=float)

similarity = compute_similarity_matrix(features)
result = PCSCS(enable_tracking=True).analyze_layer(
    similarity, layer_name="synthetic", sample_labels=list("abcde"), n_steps=200
)

print(f"convergence_threshold={result.convergence_threshold:.3f}")
print(f"critical_threshold={result.critical_threshold:.3f}")
print(f"critical_rate={result.critical_rate:.3f}")
print(f"initial_components={int(result.num_classes[0])}")
print(f"final_components={int(result.num_classes[-1])}")
print(f"merge_events={len(result.tracking_data['merge_events'])}")
