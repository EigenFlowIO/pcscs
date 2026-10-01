"""Sample tracking, merge events, and hierarchical tree construction for PCSCS."""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


class SampleTracker:
    """Track sample clustering evolution and construct hierarchical trees."""
    
    def __init__(self):
        """Initialize sample tracker."""
        self.sample_history = {}
        self.merge_events = []
        self.threshold_snapshots = []
    
    def track_clustering_evolution(self, similarity_matrix: np.ndarray,
                                 thresholds: np.ndarray,
                                 sample_labels: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Track complete clustering evolution across thresholds.
        
        Args:
            similarity_matrix: Square similarity matrix.
            thresholds: Array of similarity thresholds to analyze.
            sample_labels: Optional labels for samples.
            
        Returns:
            Dictionary containing sample history, merge events, and tree data.
        """
        n_samples = similarity_matrix.shape[0]
        sample_history = {i: [] for i in range(n_samples)}
        all_merge_events = []
        threshold_snapshots = []
        
        prev_components = None
        
        for step, threshold in enumerate(thresholds):
            # Get current clustering
            curr_components = self._find_connected_components(similarity_matrix, threshold)
            
            # Record cluster membership for each sample
            sample_to_cluster = {}
            for cluster_id, cluster_members in enumerate(curr_components):
                for sample_id in cluster_members:
                    sample_to_cluster[sample_id] = cluster_id
                    
                    # Record in sample history
                    sample_history[sample_id].append({
                        'step': step,
                        'threshold': threshold,
                        'cluster_id': cluster_id,
                        'cluster_size': len(cluster_members),
                        'cluster_members': cluster_members.copy(),
                        'co_clustered_with': [s for s in cluster_members if s != sample_id]
                    })
            
            # Detect merge events
            if prev_components is not None:
                merge_events = self._detect_merge_events(
                    prev_components, curr_components, threshold
                )
                all_merge_events.extend(merge_events)
            
            # Store snapshot
            threshold_snapshots.append({
                'threshold': threshold,
                'step': step,
                'num_clusters': len(curr_components),
                'clusters': curr_components,
                'sample_to_cluster': sample_to_cluster.copy()
            })
            
            prev_components = curr_components
        
        # Build hierarchical tree
        tree_data = self._build_clustering_tree(
            sample_history, all_merge_events, threshold_snapshots, sample_labels
        )
        
        return {
            'sample_history': sample_history,
            'merge_events': all_merge_events,
            'threshold_snapshots': threshold_snapshots,
            'tree_data': tree_data
        }
    
    def _find_connected_components(self, similarity_matrix: np.ndarray,
                                 threshold: float) -> List[List[int]]:
        """Find connected components at given threshold."""
        n = similarity_matrix.shape[0]
        visited = set()
        components = []
        
        def dfs(node: int, component: List[int]) -> None:
            visited.add(node)
            component.append(node)
            for neighbor in range(n):
                if (neighbor not in visited and 
                    similarity_matrix[node, neighbor] > threshold):
                    dfs(neighbor, component)
        
        for i in range(n):
            if i not in visited:
                component = []
                dfs(i, component)
                components.append(component)
                
        return components
    
    def _detect_merge_events(self, prev_components: List[List[int]],
                           curr_components: List[List[int]],
                           threshold: float) -> List[Dict[str, Any]]:
        """Detect cluster merge events between consecutive threshold steps."""
        merge_events = []
        
        if not prev_components:
            return merge_events
        
        # Create mapping of samples to their previous cluster ID
        prev_sample_to_cluster = {}
        for cluster_id, cluster in enumerate(prev_components):
            for sample in cluster:
                prev_sample_to_cluster[sample] = cluster_id
        
        # For each current cluster, find which previous clusters contributed
        for curr_cluster_id, curr_cluster in enumerate(curr_components):
            # Find all previous clusters that have samples in this current cluster
            contributing_prev_clusters = set()
            for sample in curr_cluster:
                if sample in prev_sample_to_cluster:
                    contributing_prev_clusters.add(prev_sample_to_cluster[sample])
            
            # If multiple previous clusters contributed, it's a merge
            if len(contributing_prev_clusters) > 1:
                merge_events.append({
                    'threshold': threshold,
                    'merged_cluster_ids': list(contributing_prev_clusters),
                    'merged_clusters': [prev_components[cid] for cid in contributing_prev_clusters],
                    'resulting_cluster': curr_cluster,
                    'resulting_cluster_id': curr_cluster_id,
                    'samples_involved': curr_cluster
                })
        
        return merge_events
    
    def _build_clustering_tree(self, sample_history: Dict[int, List[Dict]],
                             merge_events: List[Dict[str, Any]],
                             threshold_snapshots: List[Dict[str, Any]],
                             sample_labels: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Build hierarchical clustering tree structure."""
        tree_structure = self._construct_tree_structure(merge_events, threshold_snapshots)
        
        return {
            'sample_history': sample_history,
            'merge_events': merge_events,
            'threshold_snapshots': threshold_snapshots,
            'sample_labels': sample_labels,
            'tree_structure': tree_structure
        }
    
    def _construct_tree_structure(self, merge_events: List[Dict[str, Any]],
                                threshold_snapshots: List[Dict[str, Any]]) -> nx.DiGraph:
        """Construct tree structure as networkx directed graph."""
        tree = nx.DiGraph()
        
        # Get number of samples and create reverse timeline
        n_samples = len(threshold_snapshots[0]['sample_to_cluster'])
        
        # Work backwards: start from convergence, split to singletons
        reversed_snapshots = list(reversed(threshold_snapshots))
        
        # Track cluster evolution
        cluster_evolution = []
        for i, snapshot in enumerate(reversed_snapshots):
            clusters = snapshot['clusters']
            threshold = snapshot['threshold']
            
            cluster_evolution.append({
                'threshold': threshold,
                'clusters': [list(cluster) for cluster in clusters],
                'step': i
            })
        
        # Build tree nodes and edges
        cluster_to_node = {}
        node_counter = 0
        
        # Create nodes for each cluster state
        for evolution_step in cluster_evolution:
            threshold = evolution_step['threshold']
            clusters = evolution_step['clusters']
            
            for cluster in clusters:
                cluster_key = tuple(sorted(cluster))
                
                if cluster_key not in cluster_to_node:
                    node_id = f"cluster_{node_counter}"
                    cluster_to_node[cluster_key] = node_id
                    node_counter += 1
                    
                    # Determine node type
                    if len(cluster) == 1:
                        node_type = 'leaf'
                        sample_id = cluster[0]
                    else:
                        node_type = 'internal'
                        sample_id = None
                    
                    tree.add_node(node_id,
                                cluster=cluster,
                                cluster_key=cluster_key,
                                threshold=threshold,
                                node_type=node_type,
                                sample_id=sample_id,
                                size=len(cluster))
        
        # Create edges by finding parent-child relationships
        cluster_keys = list(cluster_to_node.keys())
        
        for child_key in cluster_keys:
            child_node = cluster_to_node[child_key]
            child_cluster = set(child_key)
            child_threshold = tree.nodes[child_node]['threshold']
            
            # Find potential parents
            potential_parents = []
            
            for parent_key in cluster_keys:
                if parent_key == child_key:
                    continue
                
                parent_cluster = set(parent_key)
                parent_threshold = tree.nodes[cluster_to_node[parent_key]]['threshold']
                
                # Parent must contain child and appear at lower threshold
                if (child_cluster.issubset(parent_cluster) and 
                    parent_threshold < child_threshold):
                    potential_parents.append((parent_key, parent_threshold))
            
            # Find most immediate parent
            if potential_parents:
                potential_parents.sort(key=lambda x: x[1], reverse=True)
                immediate_parent_key = potential_parents[0][0]
                parent_node = cluster_to_node[immediate_parent_key]
                
                # Verify direct split
                parent_cluster = set(immediate_parent_key)
                is_direct = True
                
                for intermediate_key in cluster_keys:
                    intermediate_cluster = set(intermediate_key)
                    intermediate_threshold = tree.nodes[cluster_to_node[intermediate_key]]['threshold']
                    
                    if (intermediate_key != child_key and intermediate_key != immediate_parent_key and
                        child_cluster.issubset(intermediate_cluster) and
                        intermediate_cluster.issubset(parent_cluster) and
                        child_threshold < intermediate_threshold < tree.nodes[parent_node]['threshold']):
                        is_direct = False
                        break
                
                if is_direct:
                    tree.add_edge(parent_node, child_node)
        
        # Order samples for visualization
        leaf_nodes = [node for node, data in tree.nodes(data=True) if data['node_type'] == 'leaf']
        sample_order = []
        
        for node in leaf_nodes:
            sample_id = tree.nodes[node]['sample_id']
            sample_order.append((sample_id, node))
        
        sample_order.sort(key=lambda x: x[0])
        
        tree.graph['sample_order'] = [sample_id for sample_id, node in sample_order]
        tree.graph['ordered_leaf_nodes'] = [node for sample_id, node in sample_order]
        
        return tree


def analyze_sample_trajectories(tracking_data: Dict[str, Any],
                              layer_name: str) -> Dict[int, Dict[str, Any]]:
    """
    Analyze individual sample clustering trajectories.
    
    Args:
        tracking_data: Sample tracking data from PCSCS analysis.
        layer_name: Layer name for reporting.
        
    Returns:
        Dictionary mapping sample IDs to trajectory statistics.
    """
    if not tracking_data or not tracking_data.get('sample_history'):
        return {}
    
    sample_history = tracking_data['sample_history']
    trajectory_stats = {}
    
    for sample_id, history in sample_history.items():
        if not history:
            continue
        
        # Find when sample first joins with others
        first_merge_threshold = None
        first_merge_size = 1
        
        for entry in history:
            if entry['cluster_size'] > 1:
                first_merge_threshold = entry['threshold']
                first_merge_size = entry['cluster_size']
                break
        
        # Find final cluster size
        final_cluster_size = history[-1]['cluster_size'] if history else 1
        
        # Count distinct clustering events
        cluster_size_changes = []
        prev_size = 1
        for entry in history:
            if entry['cluster_size'] != prev_size:
                cluster_size_changes.append({
                    'threshold': entry['threshold'],
                    'from_size': prev_size,
                    'to_size': entry['cluster_size']
                })
                prev_size = entry['cluster_size']
        
        trajectory_stats[sample_id] = {
            'first_merge_threshold': first_merge_threshold,
            'first_merge_size': first_merge_size,
            'final_cluster_size': final_cluster_size,
            'clustering_events': len(cluster_size_changes),
            'size_changes': cluster_size_changes
        }
    
    return trajectory_stats


def find_most_similar_samples(tracking_data: Dict[str, Any],
                            n_pairs: int = 5) -> List[Dict[str, Any]]:
    """
    Find pairs of samples that merge at highest thresholds.
    
    Args:
        tracking_data: Sample tracking data from PCSCS analysis.
        n_pairs: Number of most similar pairs to return.
        
    Returns:
        List of dictionaries containing merge information for most similar pairs.
    """
    if not tracking_data or not tracking_data.get('merge_events'):
        return []
    
    merge_events = tracking_data['merge_events']
    
    # Find merge events involving exactly 2 samples
    pairwise_merges = []
    for event in merge_events:
        if len(event['samples_involved']) == 2:
            pairwise_merges.append({
                'threshold': event['threshold'],
                'sample_pair': tuple(sorted(event['samples_involved'])),
                'samples': event['samples_involved']
            })
    
    # Sort by threshold (highest first = most similar)
    pairwise_merges.sort(key=lambda x: x['threshold'], reverse=True)
    
    return pairwise_merges[:n_pairs]