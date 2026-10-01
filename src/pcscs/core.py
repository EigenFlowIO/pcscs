"""Core PCSCS algorithm implementation (domain-agnostic)."""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from .analysis import fit_sigmoid, find_critical_threshold
from .tracking import SampleTracker
from .utils import compute_similarity_matrix


class PCSCSResults:
    """Container for PCSCS analysis results."""
    
    def __init__(self):
        self.layer_name: Optional[str] = None
        self.convergence_threshold: Optional[float] = None
        self.critical_threshold: Optional[float] = None
        self.critical_rate: Optional[float] = None
        self.thresholds: Optional[np.ndarray] = None
        self.num_classes: Optional[np.ndarray] = None
        self.smooth_thresholds: Optional[np.ndarray] = None
        self.smooth_classes: Optional[np.ndarray] = None
        self.sigmoid_params: Optional[np.ndarray] = None
        self.fit_successful: bool = False
        self.derivative: Optional[np.ndarray] = None
        self.tracking_data: Optional[Dict[str, Any]] = None


class PCSCS:
    """
    Progressive Cosine Similarity Classification Sifting analyzer.
    
    Analyzes representations through systematic exploration of clustering 
    behavior across progressive similarity thresholds.
    """
    
    def __init__(self, enable_tracking: bool = True):
        """
        Initialize PCSCS analyzer.
        
        Args:
            enable_tracking: Whether to enable sample tracking and merge events.
        """
        self.enable_tracking = enable_tracking
        self.results: Dict[str, PCSCSResults] = {}
        if enable_tracking:
            self.tracker = SampleTracker()
    
    def find_convergence_threshold(self, similarity_matrix: np.ndarray, 
                                 start_threshold: float = 0.999, 
                                 step_size: float = 0.02) -> float:
        """
        Find threshold where all samples form single connected component.
        
        Args:
            similarity_matrix: Square similarity matrix.
            start_threshold: Starting similarity threshold.
            step_size: Threshold decrement step size.
            
        Returns:
            Convergence threshold value.
        """
        n_samples = similarity_matrix.shape[0]
        threshold = start_threshold
        
        while threshold > 0.0:
            components = self._find_connected_components(similarity_matrix, threshold)
            
            if len(components) == 1 and len(components[0]) == n_samples:
                return threshold
                
            threshold -= step_size
            
        return threshold
    
    def _find_connected_components(self, similarity_matrix: np.ndarray, 
                                 threshold: float) -> List[List[int]]:
        """
        Find connected components at given similarity threshold.
        
        Args:
            similarity_matrix: Square similarity matrix.
            threshold: Similarity threshold for connectivity.
            
        Returns:
            List of connected components, each as list of sample indices.
        """
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
    
    def progressive_sift(self, similarity_matrix: np.ndarray,
                        convergence_threshold: float,
                        n_steps: int = 1000,
                        sample_labels: Optional[List[Any]] = None) -> tuple:
        """
        Perform progressive similarity sifting with optional sample tracking.
        
        Args:
            similarity_matrix: Square similarity matrix.
            convergence_threshold: Threshold where all samples converge.
            n_steps: Number of threshold steps.
            sample_labels: Optional labels for samples.
            
        Returns:
            Tuple of (thresholds, num_classes, tracking_data).
        """
        thresholds = np.linspace(1.0, convergence_threshold, n_steps)
        num_classes_sequence = []
        tracking_data = None
        
        if self.enable_tracking:
            tracking_data = self.tracker.track_clustering_evolution(
                similarity_matrix, thresholds, sample_labels
            )
            num_classes_sequence = [
                snapshot['num_clusters'] 
                for snapshot in tracking_data['threshold_snapshots']
            ]
        else:
            for threshold in thresholds:
                components = self._find_connected_components(similarity_matrix, threshold)
                num_classes_sequence.append(len(components))
                
        return thresholds, np.array(num_classes_sequence), tracking_data
    
    def analyze_layer(self, similarity_matrix: np.ndarray,
                     layer_name: str = "Layer",
                     sample_labels: Optional[List[Any]] = None,
                     n_steps: int = 1000) -> PCSCSResults:
        """
        Perform complete PCSCS analysis on single layer.
        
        Args:
            similarity_matrix: Square cosine similarity matrix.
            layer_name: Name identifier for this layer.
            sample_labels: Optional labels for samples.
            n_steps: Number of threshold steps for analysis.
            
        Returns:
            PCSCSResults object containing all analysis results.
        """
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("Similarity matrix must be square")
        
        if similarity_matrix.shape[0] < 2:
            raise ValueError("Need at least 2 samples for analysis")
        
        results = PCSCSResults()
        results.layer_name = layer_name
        
        # Step 1: Find convergence threshold
        convergence_threshold = self.find_convergence_threshold(similarity_matrix)
        results.convergence_threshold = convergence_threshold
        
        # Step 2: Progressive sifting
        thresholds, num_classes, tracking_data = self.progressive_sift(
            similarity_matrix, convergence_threshold, n_steps, sample_labels
        )
        
        results.thresholds = thresholds
        results.num_classes = num_classes
        results.tracking_data = tracking_data
        
        # Step 3: Sigmoid fitting
        try:
            smooth_thresholds, smooth_classes, sigmoid_params, fit_success = fit_sigmoid(
                thresholds, num_classes
            )
            results.smooth_thresholds = smooth_thresholds
            results.smooth_classes = smooth_classes
            results.sigmoid_params = sigmoid_params
            results.fit_successful = fit_success
        except Exception:
            # Fallback to linear interpolation
            results.smooth_thresholds = np.linspace(thresholds[0], thresholds[-1], 1000)
            results.smooth_classes = np.interp(
                results.smooth_thresholds, thresholds, num_classes
            )
            results.fit_successful = False
        
        # Step 4: Find critical threshold
        critical_threshold, critical_rate, derivative = find_critical_threshold(
            results.smooth_thresholds, results.smooth_classes, results.sigmoid_params
        )
        
        results.critical_threshold = critical_threshold
        results.critical_rate = critical_rate
        results.derivative = derivative
        
        # Store results
        self.results[layer_name] = results
        
        return results
    
    def analyze_layers(self, layer_features: Dict[str, Dict[str, Any]],
                      sample_labels: Optional[List[Any]] = None,
                      n_steps: int = 1000) -> Dict[str, PCSCSResults]:
        """
        Analyze multiple layers with PCSCS.
        
        Args:
            layer_features: Dictionary mapping layer names to feature data.
            sample_labels: Optional labels for samples.
            n_steps: Number of threshold steps for analysis.
            
        Returns:
            Dictionary mapping layer names to PCSCSResults.
        """
        results = {}
        
        for layer_name, layer_data in layer_features.items():
            if 'similarity_matrix' in layer_data:
                similarity_matrix = layer_data['similarity_matrix']
            else:
                # Compute similarity matrix from features
                features = layer_data.get('features')
                if features is None:
                    raise ValueError(f"No similarity matrix or features found for {layer_name}")
                similarity_matrix = compute_similarity_matrix(features)
            
            layer_results = self.analyze_layer(
                similarity_matrix, layer_name, sample_labels, n_steps
            )
            results[layer_name] = layer_results
            
        return results