import numpy as np

from pcscs.hierarchy import (
    build_filtration_linkage,
    components_from_filtration_linkage,
    threshold_graph_components,
    verify_filtration_linkage,
)


def test_filtration_linkage_matches_threshold_graph_partitions():
    sim = np.array([
        [1.00, .95, .30, .20],
        [.95, 1.00, .40, .25],
        [.30, .40, 1.00, .80],
        [.20, .25, .80, 1.00],
    ])
    Z = build_filtration_linkage(sim)
    assert Z.shape == (3, 4)
    thresholds = [.97, .90, .75, .39, .29, .10]
    verify_filtration_linkage(sim, Z, thresholds)
    for theta in thresholds:
        assert components_from_filtration_linkage(Z, 4, theta) == threshold_graph_components(sim, theta)
