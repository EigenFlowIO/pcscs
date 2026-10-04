"""Filtration-faithful hierarchy utilities for PCSCS threshold graphs.

The PCSCS graph at threshold ``theta`` contains an undirected edge ``(i, j)``
when cosine similarity ``S[i, j] > theta``.  As theta decreases, connected
components can only merge.  This module converts that exact edge-addition
filtration into a SciPy-compatible binary linkage matrix without introducing
an independent clustering objective.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
from scipy.cluster.hierarchy import dendrogram

from .utils import validate_similarity_matrix


def build_filtration_linkage(similarity_matrix: np.ndarray) -> np.ndarray:
    """Build a binary linkage representation of the PCSCS connectivity filtration.

    Edges are processed in descending cosine similarity using Kruskal-style
    union-find.  Every union of two previously disconnected components becomes
    one linkage row.  The row distance is ``1 - similarity``.  Equal-similarity
    multiway merges are represented by an arbitrary binary refinement at the
    same distance, which preserves every threshold-graph component partition.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples - 1, 4)`` with SciPy linkage semantics.
    """
    sim = np.asarray(similarity_matrix, dtype=float)
    validate_similarity_matrix(sim)
    n = sim.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples")

    iu = np.triu_indices(n, k=1)
    vals = sim[iu]
    # Stable descending sort gives deterministic tie refinement.
    order = np.argsort(-vals, kind="mergesort")

    parent = list(range(2 * n - 1))
    size = [1] * (2 * n - 1)
    cluster_id = list(range(2 * n - 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rows: List[List[float]] = []
    next_id = n

    for pos in order:
        i = int(iu[0][pos])
        j = int(iu[1][pos])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        ci, cj = cluster_id[ri], cluster_id[rj]
        merged_size = size[ri] + size[rj]
        similarity = float(vals[pos])
        rows.append([float(ci), float(cj), float(1.0 - similarity), float(merged_size)])

        new_root = next_id
        parent[ri] = new_root
        parent[rj] = new_root
        parent[new_root] = new_root
        size[new_root] = merged_size
        cluster_id[new_root] = next_id
        next_id += 1
        if len(rows) == n - 1:
            break

    if len(rows) != n - 1:
        raise RuntimeError(f"Expected {n - 1} filtration merges, obtained {len(rows)}")
    return np.asarray(rows, dtype=float)


def components_from_filtration_linkage(
    linkage_matrix: np.ndarray,
    n_samples: int,
    threshold: float,
) -> list[list[int]]:
    """Return PCSCS connected components implied by a filtration linkage cut.

    The strict PCSCS edge rule is preserved: a merge is active only when the
    merge similarity is strictly greater than ``threshold``.
    """
    Z = np.asarray(linkage_matrix, dtype=float)
    members: dict[int, set[int]] = {i: {i} for i in range(n_samples)}
    active: dict[int, set[int]] = {i: {i} for i in range(n_samples)}

    for row_idx, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        sim = 1.0 - float(row[2])
        new_id = n_samples + row_idx
        merged = set(members[left]) | set(members[right])
        members[new_id] = merged
        if sim > threshold:
            # Remove the currently-active descendants contained in this merge.
            for cid in list(active):
                if active[cid].issubset(merged):
                    del active[cid]
            active[new_id] = merged

    comps = [sorted(v) for v in active.values()]
    comps.sort(key=lambda x: (x[0], len(x)))
    return comps


def threshold_graph_components(similarity_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """Reference connected components for the PCSCS ``similarity > threshold`` graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    sim = np.asarray(similarity_matrix, dtype=float)
    adjacency = sim > float(threshold)
    np.fill_diagonal(adjacency, False)
    n_comp, labels = connected_components(csr_matrix(adjacency), directed=False)
    comps = [np.flatnonzero(labels == k).astype(int).tolist() for k in range(n_comp)]
    comps.sort(key=lambda x: (x[0], len(x)))
    return comps


def verify_filtration_linkage(
    similarity_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    thresholds: Iterable[float],
) -> None:
    """Raise if any linkage cut disagrees with the PCSCS threshold graph."""
    n = int(np.asarray(similarity_matrix).shape[0])
    for theta in thresholds:
        a = threshold_graph_components(similarity_matrix, float(theta))
        b = components_from_filtration_linkage(linkage_matrix, n, float(theta))
        if a != b:
            raise AssertionError(f"Filtration linkage mismatch at threshold {theta}")


def plot_filtration_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str] | None = None,
    threshold: float | None = None,
    ax=None,
    leaf_font_size: float = 5.0,
):
    """Plot the filtration-faithful linkage using cosine-similarity tick labels.

    This is a view of the PCSCS connectivity filtration itself, not average-link
    or another independent hierarchical clustering procedure.
    """
    import matplotlib.pyplot as plt

    Z = np.asarray(linkage_matrix, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 8))
    color_threshold = None if threshold is None else 1.0 - float(threshold)
    out = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=leaf_font_size,
        color_threshold=color_threshold,
        ax=ax,
    )
    if threshold is not None:
        ax.axhline(1.0 - float(threshold), linestyle="--", linewidth=1.2)

    # Keep the actual coordinate as distance but label the axis in similarity,
    # which is the natural PCSCS threshold coordinate.
    ticks = ax.get_yticks()
    ax.set_yticklabels([f"{1.0 - t:.3f}" for t in ticks])
    ax.set_ylabel("Cosine similarity at component merger")
    ax.set_xlabel("Probe samples")
    return out
