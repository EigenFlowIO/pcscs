"""Visualization functions for PCSCS analysis results (domain-agnostic)."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

from .core import PCSCSResults
from .tracking import analyze_sample_trajectories, find_most_similar_samples


def plot_layer_analysis(results: PCSCSResults,
                       figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot PCSCS analysis for single layer with sigmoid fit and derivative.
    
    Args:
        results: PCSCS analysis results for a layer.
        figsize: Figure size as (width, height).
    """
    if results.thresholds is None:
        raise ValueError("No analysis results found")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Raw data and sigmoid fit
    ax1.scatter(results.thresholds, results.num_classes,
                alpha=0.4, s=2, color='blue', label='Raw data')
    
    if results.smooth_thresholds is not None:
        fit_label = 'Sigmoid fit' if results.fit_successful else 'Linear interpolation'
        ax1.plot(results.smooth_thresholds, results.smooth_classes,
                'red', linewidth=2, label=fit_label)
    
    if results.critical_threshold is not None:
        ax1.axvline(results.critical_threshold, color='orange',
                    linestyle='--', linewidth=2,
                    label=f'Critical: {results.critical_threshold:.3f}')
    
    ax1.set_xlabel('Cosine Similarity Threshold')
    ax1.set_ylabel('Number of Classes')
    ax1.set_title(f'{results.layer_name}: PCSCS Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Derivative analysis
    if results.derivative is not None:
        ax2.plot(results.smooth_thresholds, results.derivative,
                'green', linewidth=2, label='Rate of increase')
        
        if results.critical_threshold is not None:
            ax2.axvline(results.critical_threshold, color='orange',
                        linestyle='--', linewidth=2,
                        label=f'Max rate: {results.critical_rate:.1f}')
        
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Cosine Similarity Threshold')
        ax2.set_ylabel('Rate of Increase')
        ax2.set_title(f'{results.layer_name}: Derivative Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_merge_events(results: PCSCSResults) -> None:
    """
    Plot merge events as scatter plot.
    
    Args:
        results: PCSCS analysis results for a layer.
    """
    if not results.tracking_data or not results.tracking_data.get('merge_events'):
        print(f"No merge events data available for {results.layer_name}")
        return
    
    merge_events = results.tracking_data['merge_events']
    merge_thresholds = [event['threshold'] for event in merge_events]
    merge_sizes = [len(event['samples_involved']) for event in merge_events]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(merge_thresholds, merge_sizes, alpha=0.6, s=30)
    plt.xlabel('Threshold')
    plt.ylabel('Merge Size (samples)')
    plt.title(f'{results.layer_name}: Merge Events')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_dendrogram(results: PCSCSResults,
                   test_threshold: Optional[float] = None,
                   similarity_scale: str = 'linear',
                   figsize: Tuple[int, int] = (16, 8)) -> Optional[Dict]:
    """
    Create dendrogram visualization for clustering tree.
    
    Args:
        results: PCSCS analysis results for a layer.
        test_threshold: Optional threshold for color coding similarity classes.
        similarity_scale: 'linear' or 'squared' scaling for y-axis.
        figsize: Figure size as (width, height).
        
    Returns:
        Dendrogram data dictionary from scipy, or None if no data available.
    """
    if (not results.tracking_data or 
        not results.tracking_data.get('merge_events')):
        print(f"No tree data available for {results.layer_name}")
        return None
    
    merge_events = results.tracking_data['merge_events']
    sample_labels = results.tracking_data.get('sample_labels')
    
    # Get similarity range from merge events
    merge_similarities = [event['threshold'] for event in merge_events]
    if not merge_similarities:
        print(f"No merge events found for {results.layer_name}")
        return None
    
    min_similarity = min(merge_similarities)
    max_similarity = max(merge_similarities)
    
    # Get number of samples
    all_samples = set()
    for event in merge_events:
        all_samples.update(event['samples_involved'])
    n_samples = len(all_samples)
    
    if n_samples < 2:
        print("Need at least 2 samples for dendrogram")
        return None
    
    # Build distance matrix
    distance_matrix = np.ones((n_samples, n_samples))
    np.fill_diagonal(distance_matrix, 0)
    
    for event in merge_events:
        samples_involved = event['samples_involved']
        threshold = event['threshold']
        distance = 1.0 - threshold
        
        for i in range(len(samples_involved)):
            for j in range(i + 1, len(samples_involved)):
                sample_i = samples_involved[i]
                sample_j = samples_involved[j]
                
                if sample_i < n_samples and sample_j < n_samples:
                    current_dist = distance_matrix[sample_i, sample_j]
                    distance_matrix[sample_i, sample_j] = min(current_dist, distance)
                    distance_matrix[sample_j, sample_i] = min(current_dist, distance)
    
    # Create linkage matrix
    condensed_distances = squareform(distance_matrix)
    Z = linkage(condensed_distances, method='average')
    
    # Transform for similarity scale
    if similarity_scale == 'squared':
        Z_transformed = Z.copy()
        distances = Z_transformed[:, 2]
        similarities = 1.0 - distances
        squared_similarities = similarities ** 2
        Z_transformed[:, 2] = 1.0 - squared_similarities
    else:
        Z_transformed = Z
    
    # Set color threshold
    if test_threshold is not None:
        if similarity_scale == 'squared':
            color_threshold = 1.0 - (test_threshold ** 2)
        else:
            color_threshold = 1.0 - test_threshold
    else:
        color_threshold = 0.7 * np.max(Z_transformed[:, 2]) if len(Z_transformed) > 0 else 0
    
    # Get dendrogram data
    temp_dn = dendrogram(Z_transformed, no_plot=True, color_threshold=color_threshold)
    
    # Get similarity classes if test threshold provided
    sample_to_class = {}
    if test_threshold is not None:
        similarity_matrix = 1.0 - distance_matrix
        sample_to_class = _get_similarity_classes(similarity_matrix, test_threshold)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors for similarity classes
    class_colors = ['black']
    if test_threshold is not None and sample_to_class:
        n_classes = len(set(sample_to_class.values()))
        cmap = cm.get_cmap('hsv')
        class_colors = [mcolors.to_hex(cmap(i / n_classes)) for i in range(n_classes)]
    
    # Draw dendrogram branches
    icoord = temp_dn['icoord']
    dcoord = temp_dn['dcoord']
    leaf_order = temp_dn['leaves']
    scipy_colors = temp_dn['color_list']
    
    for i, (xs, ys) in enumerate(zip(icoord, dcoord)):
        original_color = scipy_colors[i] if i < len(scipy_colors) else 'C0'
        branch_color = _determine_branch_color(
            xs, ys, original_color, test_threshold, sample_to_class, 
            leaf_order, class_colors
        )
        ax.plot(xs, ys, color=branch_color, linewidth=2)
    
    # Create labels
    labels = _create_sample_labels(leaf_order, sample_labels)
    
    # Set x-axis
    x_positions = range(0, len(leaf_order) * 10, 10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    
    # Color x-axis labels
    if test_threshold is not None and sample_to_class:
        for label_obj, leaf_idx in zip(ax.get_xticklabels(), leaf_order):
            if leaf_idx in sample_to_class:
                class_id = sample_to_class[leaf_idx]
                label_obj.set_color(class_colors[class_id])
    
    # Add horizontal line at test threshold
    if test_threshold is not None:
        test_distance = (1.0 - (test_threshold ** 2) if similarity_scale == 'squared' 
                        else 1.0 - test_threshold)
        ax.axhline(y=test_distance, color='grey', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set y-axis
    if similarity_scale == 'squared':
        min_distance = 1.0 - (max_similarity ** 2)
        max_distance = 1.0 - (min_similarity ** 2)
        ax.set_ylabel('Cosine Similarity²', fontsize=12)
    else:
        min_distance = 1.0 - max_similarity
        max_distance = 1.0 - min_similarity
        ax.set_ylabel('Cosine Similarity', fontsize=12)
    
    ax.set_ylim(min_distance - 0.01, max_distance + 0.01)
    
    # Create y-axis tick labels
    n_ticks = 8
    distance_ticks = np.linspace(min_distance, max_distance, n_ticks)
    similarities = 1.0 - distance_ticks
    ax.set_yticks(distance_ticks)
    ax.set_yticklabels([f'{sim:.3f}' for sim in similarities])
    
    ax.set_title(f'PCSCS Dendrogram: {results.layer_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Samples', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return temp_dn


def get_similarity_class_partitions(results: PCSCSResults,
                                  threshold: float) -> List[List[int]]:
    """
    Get similarity class partitions as lists of sample indices.
    
    Args:
        results: PCSCS analysis results for a layer.
        threshold: Cosine similarity threshold.
        
    Returns:
        List of lists, each containing sample indices in same similarity class.
        Sorted by class size (largest first).
    """
    if (not results.tracking_data or 
        not results.tracking_data.get('threshold_snapshots')):
        return []
    
    threshold_snapshots = results.tracking_data['threshold_snapshots']
    
    # Find closest snapshot to requested threshold
    closest_snapshot = None
    min_diff = float('inf')
    
    for snapshot in threshold_snapshots:
        diff = abs(snapshot['threshold'] - threshold)
        if diff < min_diff:
            min_diff = diff
            closest_snapshot = snapshot
    
    if closest_snapshot is None:
        return []
    
    clusters = closest_snapshot['clusters']
    similarity_classes = [list(cluster) for cluster in clusters]
    
    # Sort by size (largest first) and then by first sample ID
    similarity_classes.sort(key=lambda x: (-len(x), min(x) if x else float('inf')))
    
    return similarity_classes


def print_class_summary(similarity_classes: List[List[int]], 
                       sample_labels: Optional[List[Any]] = None,
                       layer_name: str = "Layer",
                       threshold: float = 0.0) -> None:
    """
    Print summary of similarity classes (domain-agnostic).
    
    Args:
        similarity_classes: List of lists of sample indices.
        sample_labels: Optional labels for samples.
        layer_name: Layer name for display.
        threshold: Threshold used for classification.
    """
    print(f"Similarity Classes for {layer_name} at threshold {threshold:.3f}")
    print(f"Found {len(similarity_classes)} classes")
    
    for class_idx, sample_indices in enumerate(similarity_classes):
        print(f"\nClass {class_idx + 1}: {len(sample_indices)} samples")
        print(f"  Sample indices: {sample_indices}")
        
        if sample_labels:
            # Show label distribution for this class
            class_labels = [sample_labels[i] for i in sample_indices if i < len(sample_labels)]
            if class_labels and isinstance(class_labels[0], dict):
                # Count by some key like 'taxon'
                key_counts = Counter([label.get('taxon', 'Unknown') for label in class_labels])
                print(f"  Label distribution: {dict(key_counts)}")


def plot_combined_layers(layer_results: Dict[str, PCSCSResults],
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Create combined plot showing all layers with critical thresholds.
    
    Args:
        layer_results: Dictionary mapping layer names to PCSCS results.
        figsize: Figure size as (width, height).
    """
    if not layer_results:
        print("No results to plot")
        return
    
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(layer_results)))
    
    for i, (layer_name, results) in enumerate(layer_results.items()):
        if results.smooth_thresholds is not None and results.smooth_classes is not None:
            plt.plot(results.smooth_thresholds, results.smooth_classes,
                    color=colors[i], linewidth=2, alpha=0.7, label=layer_name)
            
            # Mark critical threshold
            if results.critical_threshold is not None:
                plt.axvline(results.critical_threshold,
                            color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
                
                # Add annotation for critical threshold
                y_at_critical = np.interp(results.critical_threshold,
                                        results.smooth_thresholds,
                                        results.smooth_classes)
                
                plt.annotate(f'{layer_name}\n{results.critical_threshold:.3f}',
                            xy=(results.critical_threshold, y_at_critical),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7,
                            bbox=dict(boxstyle='round,pad=0.2', 
                                     facecolor=colors[i], alpha=0.3))
    
    plt.xlabel('Cosine Similarity Threshold')
    plt.ylabel('Number of Classes')
    plt.title('PCSCS Analysis: All Layers\nSigmoid Fits with Critical Thresholds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Legacy function for backwards compatibility
def get_similarity_classes(results: PCSCSResults, threshold: float) -> List[List[int]]:
    """Legacy function - use get_similarity_class_partitions instead."""
    return get_similarity_class_partitions(results, threshold)


def _get_similarity_classes(similarity_matrix: np.ndarray, 
                          threshold: float) -> Dict[int, int]:
    """Get similarity classes at threshold using DFS."""
    n_samples = similarity_matrix.shape[0]
    visited = set()
    sample_to_class = {}
    class_id = 0
    
    def dfs(node: int, current_class: int) -> None:
        visited.add(node)
        sample_to_class[node] = current_class
        for neighbor in range(n_samples):
            if neighbor not in visited and similarity_matrix[node, neighbor] > threshold:
                dfs(neighbor, current_class)
    
    for i in range(n_samples):
        if i not in visited:
            dfs(i, class_id)
            class_id += 1
    
    return sample_to_class


def _determine_branch_color(xs: List[float], ys: List[float], 
                          original_color: str, test_threshold: Optional[float],
                          sample_to_class: Dict[int, int], leaf_order: List[int],
                          class_colors: List[str]) -> str:
    """Determine color for dendrogram branch based on similarity classes."""
    if test_threshold is None or not sample_to_class:
        return original_color
    
    # Only change branches that scipy colored as 'C0'
    if original_color != 'C0':
        return original_color
    
    # For leaf branches (connecting directly to samples)
    if min(ys) == 0:
        leaf_x = xs[0] if ys[0] == 0 else xs[-1]
        leaf_idx_in_order = int(leaf_x / 10)
        if leaf_idx_in_order < len(leaf_order):
            actual_leaf = leaf_order[leaf_idx_in_order]
            if actual_leaf in sample_to_class:
                class_id = sample_to_class[actual_leaf]
                return class_colors[class_id]
    else:
        # Internal branch - find all leaves under this branch
        x_min, x_max = min(xs), max(xs)
        connected_leaves = set()
        for j, leaf_idx in enumerate(leaf_order):
            leaf_x = j * 10
            if x_min <= leaf_x <= x_max:
                connected_leaves.add(leaf_idx)
        
        # Get all classes represented
        classes = set()
        for leaf_sample in connected_leaves:
            if leaf_sample in sample_to_class:
                classes.add(sample_to_class[leaf_sample])
        
        # If all leaves belong to same class, use that color
        if len(classes) == 1:
            class_id = list(classes)[0]
            return class_colors[class_id]
    
    return original_color


def _create_sample_labels(leaf_order: List[int], 
                        sample_labels: Optional[List[Any]]) -> List[str]:
    """Create labels for dendrogram x-axis."""
    labels = []
    for leaf_idx in leaf_order:
        if sample_labels and leaf_idx < len(sample_labels) and sample_labels[leaf_idx]:
            label_data = sample_labels[leaf_idx]
            if isinstance(label_data, dict):
                label_text = label_data.get('label', label_data.get('name', f'S{leaf_idx}'))
                if len(str(label_text)) > 10:
                    label_text = str(label_text)[:8] + '..'
                labels.append(f"S{leaf_idx}:{label_text}")
            else:
                labels.append(f"S{leaf_idx}:{str(label_data)[:8]}")
        else:
            labels.append(f"S{leaf_idx}")
    return labels