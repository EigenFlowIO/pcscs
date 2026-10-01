"""Spectral analysis module for PCSCS - graph theoretical analysis of similarity structures."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .utils import validate_similarity_matrix


class SpectralResults:
    """Container for spectral analysis results."""
    
    def __init__(self):
        self.threshold: Optional[float] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.algebraic_connectivity: Optional[float] = None
        self.spectral_gap: Optional[float] = None
        self.effective_dimension: Optional[int] = None
        self.spectral_entropy: Optional[float] = None
        self.n_components: Optional[int] = None


class DynamicSpectralResults:
    """Container for dynamic spectral analysis results."""
    
    def __init__(self):
        self.thresholds: Optional[np.ndarray] = None
        self.eigenvalue_evolution: Optional[np.ndarray] = None
        self.algebraic_connectivity_evolution: Optional[np.ndarray] = None
        self.spectral_gap_evolution: Optional[np.ndarray] = None
        self.effective_dimension_evolution: Optional[np.ndarray] = None
        self.n_components_evolution: Optional[np.ndarray] = None
        self.percolation_threshold: Optional[float] = None
        self.spectral_critical_thresholds: Optional[List[float]] = None


class SpectralAnalyzer:
    """
    Spectral analysis of similarity matrices using graph Laplacian eigenvalues.
    
    Provides both static analysis (single threshold) and dynamic analysis 
    (threshold evolution) of spectral properties.
    """
    
    def __init__(self, similarity_matrix: np.ndarray):
        """
        Initialize spectral analyzer.
        
        Args:
            similarity_matrix: Square similarity matrix for analysis.
        """
        validate_similarity_matrix(similarity_matrix)
        self.similarity_matrix = similarity_matrix
        self.n_samples = similarity_matrix.shape[0]
    
    def compute_graph_laplacian(self, threshold: Optional[float] = None, 
                               normalized: bool = True) -> np.ndarray:
        """
        Compute graph Laplacian from similarity matrix.
        
        Args:
            threshold: Optional threshold for binary adjacency. If None, uses weighted graph.
            normalized: Whether to compute normalized Laplacian.
            
        Returns:
            Laplacian matrix.
        """
        if threshold is not None:
            # Binary adjacency matrix
            adjacency = (self.similarity_matrix > threshold).astype(float)
            np.fill_diagonal(adjacency, 0)  # Remove self-loops
        else:
            # Weighted adjacency matrix
            adjacency = self.similarity_matrix.copy()
            np.fill_diagonal(adjacency, 0)  # Remove self-loops
        
        # Compute degree matrix
        degrees = np.sum(adjacency, axis=1)
        degree_matrix = np.diag(degrees)
        
        # Compute Laplacian
        laplacian = degree_matrix - adjacency
        
        if normalized and np.any(degrees > 0):
            # Normalized Laplacian: L̃ = D^(-1/2) L D^(-1/2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                deg_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
            
            deg_inv_sqrt_matrix = np.diag(deg_inv_sqrt)
            laplacian = deg_inv_sqrt_matrix @ laplacian @ deg_inv_sqrt_matrix
        
        return laplacian
    
    def compute_eigenvalues(self, laplacian: np.ndarray, 
                          k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of Laplacian matrix.
        
        Args:
            laplacian: Laplacian matrix.
            k: Number of smallest eigenvalues to compute. If None, computes all.
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in ascending order.
        """
        if k is not None and k < self.n_samples:
            # Use sparse eigensolver for partial eigendecomposition
            try:
                eigenvalues, eigenvectors = eigsh(csr_matrix(laplacian), 
                                                 k=k, which='SM', sigma=0)
            except:
                # Fallback to dense solver
                eigenvalues, eigenvectors = eigh(laplacian)
                eigenvalues = eigenvalues[:k]
                eigenvectors = eigenvectors[:, :k]
        else:
            # Dense eigendecomposition
            eigenvalues, eigenvectors = eigh(laplacian)
        
        # Sort eigenvalues and eigenvectors
        sort_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        return eigenvalues, eigenvectors
    
    def analyze_spectral_properties(self, eigenvalues: np.ndarray) -> Dict[str, float]:
        """
        Analyze spectral properties from eigenvalues.
        
        Args:
            eigenvalues: Array of eigenvalues in ascending order.
            
        Returns:
            Dictionary of spectral properties.
        """
        properties = {}
        
        # Count zero eigenvalues (connected components)
        zero_threshold = 1e-10
        n_zero = np.sum(eigenvalues < zero_threshold)
        properties['n_components'] = n_zero
        
        # Algebraic connectivity (second smallest eigenvalue)
        if len(eigenvalues) > 1:
            properties['algebraic_connectivity'] = eigenvalues[1]
        else:
            properties['algebraic_connectivity'] = 0.0
        
        # Spectral gap
        if len(eigenvalues) > 1:
            properties['spectral_gap'] = eigenvalues[1] - eigenvalues[0]
        else:
            properties['spectral_gap'] = 0.0
        
        # Effective dimension (number of "significant" eigenvalues)
        if np.max(eigenvalues) > zero_threshold:
            normalized_eigenvalues = eigenvalues / np.max(eigenvalues)
            significant_threshold = 0.01  # 1% of maximum
            properties['effective_dimension'] = np.sum(normalized_eigenvalues > significant_threshold)
        else:
            properties['effective_dimension'] = 0
        
        # Spectral entropy (diversity measure)
        positive_eigenvalues = eigenvalues[eigenvalues > zero_threshold]
        if len(positive_eigenvalues) > 1:
            # Normalize eigenvalues to form probability distribution
            p = positive_eigenvalues / np.sum(positive_eigenvalues)
            properties['spectral_entropy'] = -np.sum(p * np.log(p + 1e-12))
        else:
            properties['spectral_entropy'] = 0.0
        
        return properties
    
    def static_spectral_analysis(self, threshold: Optional[float] = None,
                                normalized: bool = True,
                                k: Optional[int] = None) -> SpectralResults:
        """
        Perform static spectral analysis at single threshold.
        
        Args:
            threshold: Similarity threshold for analysis. If None, uses weighted graph.
            normalized: Whether to use normalized Laplacian.
            k: Number of eigenvalues to compute. If None, computes all.
            
        Returns:
            SpectralResults object with analysis results.
        """
        results = SpectralResults()
        results.threshold = threshold
        
        # Compute Laplacian
        laplacian = self.compute_graph_laplacian(threshold, normalized)
        
        # Compute eigenvalues
        eigenvalues, eigenvectors = self.compute_eigenvalues(laplacian, k)
        results.eigenvalues = eigenvalues
        results.eigenvectors = eigenvectors
        
        # Analyze spectral properties
        properties = self.analyze_spectral_properties(eigenvalues)
        results.algebraic_connectivity = properties['algebraic_connectivity']
        results.spectral_gap = properties['spectral_gap']
        results.effective_dimension = properties['effective_dimension']
        results.spectral_entropy = properties['spectral_entropy']
        # Count connected components from the graph itself rather than from a
        # potentially truncated eigenvalue spectrum. The latter caps the count at k.
        if threshold is not None:
            adjacency = (self.similarity_matrix > threshold).astype(int)
        else:
            adjacency = (self.similarity_matrix != 0).astype(int)
        np.fill_diagonal(adjacency, 0)
        results.n_components = int(connected_components(csr_matrix(adjacency), directed=False, return_labels=False))
        
        return results
    
    def dynamic_spectral_analysis(self, thresholds: np.ndarray,
                                 normalized: bool = True,
                                 k: Optional[int] = 10) -> DynamicSpectralResults:
        """
        Perform dynamic spectral analysis across threshold range.
        
        Args:
            thresholds: Array of similarity thresholds to analyze.
            normalized: Whether to use normalized Laplacian.
            k: Number of eigenvalues to track. If None, computes all.
            
        Returns:
            DynamicSpectralResults object with evolution data.
        """
        results = DynamicSpectralResults()
        results.thresholds = thresholds
        
        if k is None:
            k = min(10, self.n_samples)  # Default to 10 eigenvalues
        
        # Initialize storage arrays
        eigenvalue_evolution = np.zeros((len(thresholds), k))
        algebraic_connectivity = np.zeros(len(thresholds))
        spectral_gaps = np.zeros(len(thresholds))
        effective_dimensions = np.zeros(len(thresholds))
        n_components = np.zeros(len(thresholds))
        
        # Analyze each threshold
        for i, threshold in enumerate(thresholds):
            try:
                analysis = self.static_spectral_analysis(threshold, normalized, k)
                
                # Store eigenvalues (pad with zeros if fewer than k)
                n_eigs = len(analysis.eigenvalues)
                eigenvalue_evolution[i, :n_eigs] = analysis.eigenvalues
                
                # Store scalar properties
                algebraic_connectivity[i] = analysis.algebraic_connectivity
                spectral_gaps[i] = analysis.spectral_gap
                effective_dimensions[i] = analysis.effective_dimension
                n_components[i] = analysis.n_components
                
            except Exception as e:
                # Handle numerical issues gracefully
                eigenvalue_evolution[i, :] = np.nan
                algebraic_connectivity[i] = np.nan
                spectral_gaps[i] = np.nan
                effective_dimensions[i] = np.nan
                n_components[i] = np.nan
        
        # Store results
        results.eigenvalue_evolution = eigenvalue_evolution
        results.algebraic_connectivity_evolution = algebraic_connectivity
        results.spectral_gap_evolution = spectral_gaps
        results.effective_dimension_evolution = effective_dimensions
        results.n_components_evolution = n_components
        
        # Find critical thresholds
        results.percolation_threshold = self._find_percolation_threshold(
            thresholds, n_components
        )
        results.spectral_critical_thresholds = self._find_spectral_critical_thresholds(
            thresholds, algebraic_connectivity, spectral_gaps
        )
        
        return results
    
    def _find_percolation_threshold(self, thresholds: np.ndarray, 
                                  n_components: np.ndarray) -> Optional[float]:
        """Find threshold where giant component emerges (percolation threshold)."""
        # Look for transition from many components to few
        if len(n_components) < 2:
            return None
        
        # Find where number of components drops significantly
        component_diff = np.diff(n_components)
        if len(component_diff) == 0:
            return None
        
        # Find largest drop in number of components
        max_drop_idx = np.argmin(component_diff)
        
        # Require significant drop (at least 20% of total samples)
        max_drop = -component_diff[max_drop_idx]
        if max_drop > 0.2 * self.n_samples:
            return thresholds[max_drop_idx]
        
        return None
    
    def _find_spectral_critical_thresholds(self, thresholds: np.ndarray,
                                         algebraic_connectivity: np.ndarray,
                                         spectral_gaps: np.ndarray) -> List[float]:
        """Find thresholds where spectral properties change dramatically."""
        critical_thresholds = []
        
        # Find peaks in algebraic connectivity derivative
        if len(algebraic_connectivity) > 2:
            # Compute derivative (rate of change)
            ac_derivative = np.gradient(algebraic_connectivity, thresholds)
            
            # Find local maxima in derivative (rapid changes)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(np.abs(ac_derivative), height=np.std(ac_derivative))
            
            # Add significant peaks as critical thresholds
            for peak_idx in peaks:
                if 0 < peak_idx < len(thresholds) - 1:  # Exclude boundaries
                    critical_thresholds.append(thresholds[peak_idx])
        
        # Find peaks in spectral gap
        if len(spectral_gaps) > 2:
            gap_derivative = np.gradient(spectral_gaps, thresholds)
            
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(np.abs(gap_derivative), height=np.std(gap_derivative))
            
            for peak_idx in peaks:
                if 0 < peak_idx < len(thresholds) - 1:
                    threshold = thresholds[peak_idx]
                    if threshold not in critical_thresholds:
                        critical_thresholds.append(threshold)
        
        return sorted(critical_thresholds)


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, 
                           title: str = "Eigenvalue Spectrum",
                           figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot eigenvalue spectrum.
    
    Args:
        eigenvalues: Array of eigenvalues.
        title: Plot title.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    
    plt.plot(range(len(eigenvalues)), eigenvalues, 'bo-', markersize=4)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Highlight spectral gap if present
    if len(eigenvalues) > 1:
        plt.axhline(y=eigenvalues[1], color='red', linestyle='--', 
                   alpha=0.7, label=f'λ₂ (Alg. Connectivity): {eigenvalues[1]:.3f}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_spectral_flow(dynamic_results: DynamicSpectralResults,
                      n_eigenvalues: int = 5,
                      figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot eigenvalue evolution across thresholds (spectral flow).
    
    Args:
        dynamic_results: Results from dynamic spectral analysis.
        n_eigenvalues: Number of eigenvalues to plot.
        figsize: Figure size.
    """
    if dynamic_results.eigenvalue_evolution is None:
        print("No eigenvalue evolution data available")
        return
    
    thresholds = dynamic_results.thresholds
    eigenvalues = dynamic_results.eigenvalue_evolution
    
    plt.figure(figsize=figsize)
    
    # Plot evolution of first n eigenvalues
    colors = plt.cm.viridis(np.linspace(0, 1, n_eigenvalues))
    
    for i in range(min(n_eigenvalues, eigenvalues.shape[1])):
        plt.plot(thresholds, eigenvalues[:, i], 
                color=colors[i], linewidth=2, label=f'λ{i}')
    
    # Mark critical thresholds
    if dynamic_results.spectral_critical_thresholds:
        for threshold in dynamic_results.spectral_critical_thresholds:
            plt.axvline(threshold, color='red', linestyle='--', alpha=0.7)
    
    # Mark percolation threshold
    if dynamic_results.percolation_threshold:
        plt.axvline(dynamic_results.percolation_threshold, color='orange', 
                   linestyle=':', linewidth=2, label='Percolation')
    
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Eigenvalue')
    plt.title('Spectral Flow: Eigenvalue Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectral_properties(dynamic_results: DynamicSpectralResults,
                            figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot evolution of key spectral properties.
    
    Args:
        dynamic_results: Results from dynamic spectral analysis.
        figsize: Figure size.
    """
    if dynamic_results.thresholds is None:
        print("No dynamic spectral data available")
        return
    
    thresholds = dynamic_results.thresholds
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Algebraic connectivity
    ax = axes[0, 0]
    ax.plot(thresholds, dynamic_results.algebraic_connectivity_evolution, 
            'b-', linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Algebraic Connectivity (λ₂)')
    ax.set_title('Algebraic Connectivity Evolution')
    ax.grid(True, alpha=0.3)
    
    # Spectral gap
    ax = axes[0, 1]
    ax.plot(thresholds, dynamic_results.spectral_gap_evolution, 
            'g-', linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Spectral Gap (λ₂ - λ₁)')
    ax.set_title('Spectral Gap Evolution')
    ax.grid(True, alpha=0.3)
    
    # Effective dimension
    ax = axes[1, 0]
    ax.plot(thresholds, dynamic_results.effective_dimension_evolution, 
            'r-', linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Effective Dimension')
    ax.set_title('Effective Dimension Evolution')
    ax.grid(True, alpha=0.3)
    
    # Number of components
    ax = axes[1, 1]
    ax.plot(thresholds, dynamic_results.n_components_evolution, 
            'm-', linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Number of Components')
    ax.set_title('Connected Components Evolution')
    ax.grid(True, alpha=0.3)
    
    # Mark critical thresholds on all plots
    for ax in axes.flat:
        if dynamic_results.spectral_critical_thresholds:
            for threshold in dynamic_results.spectral_critical_thresholds:
                ax.axvline(threshold, color='red', linestyle='--', alpha=0.5)
        
        if dynamic_results.percolation_threshold:
            ax.axvline(dynamic_results.percolation_threshold, color='orange', 
                      linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def compare_pcscs_spectral_thresholds(pcscs_results, spectral_results) -> Dict[str, Any]:
    """
    Compare critical thresholds from PCSCS and spectral analysis.
    
    Args:
        pcscs_results: PCSCSResults object.
        spectral_results: DynamicSpectralResults object.
        
    Returns:
        Dictionary with comparison results.
    """
    comparison = {
        'pcscs_critical_threshold': pcscs_results.critical_threshold,
        'spectral_critical_thresholds': spectral_results.spectral_critical_thresholds,
        'percolation_threshold': spectral_results.percolation_threshold,
        'agreement_analysis': {}
    }
    
    if (pcscs_results.critical_threshold is not None and 
        spectral_results.spectral_critical_thresholds):
        
        pcscs_threshold = pcscs_results.critical_threshold
        spectral_thresholds = spectral_results.spectral_critical_thresholds
        
        # Find closest spectral threshold to PCSCS threshold
        distances = [abs(st - pcscs_threshold) for st in spectral_thresholds]
        closest_spectral = spectral_thresholds[np.argmin(distances)]
        min_distance = min(distances)
        
        comparison['agreement_analysis'] = {
            'closest_spectral_threshold': closest_spectral,
            'threshold_difference': min_distance,
            'relative_difference': min_distance / pcscs_threshold if pcscs_threshold > 0 else float('inf'),
            'agreement_level': 'high' if min_distance < 0.05 else 'moderate' if min_distance < 0.1 else 'low'
        }
    
    return comparison


def print_spectral_summary(dynamic_results: DynamicSpectralResults,
                          layer_name: str = "Layer") -> None:
    """
    Print summary of spectral analysis results.
    
    Args:
        dynamic_results: Dynamic spectral analysis results.
        layer_name: Name of layer for display.
    """
    print(f"\nSpectral Analysis Summary: {layer_name}")
    print("-" * 50)
    
    if dynamic_results.percolation_threshold:
        print(f"Percolation threshold: {dynamic_results.percolation_threshold:.3f}")
    else:
        print("Percolation threshold: Not detected")
    
    if dynamic_results.spectral_critical_thresholds:
        print(f"Spectral critical thresholds: {[f'{t:.3f}' for t in dynamic_results.spectral_critical_thresholds]}")
    else:
        print("Spectral critical thresholds: None detected")
    
    # Summary statistics at key thresholds
    if (dynamic_results.thresholds is not None and 
        dynamic_results.algebraic_connectivity_evolution is not None):
        
        mid_idx = len(dynamic_results.thresholds) // 2
        mid_threshold = dynamic_results.thresholds[mid_idx]
        mid_connectivity = dynamic_results.algebraic_connectivity_evolution[mid_idx]
        
        print(f"Mid-range connectivity (θ={mid_threshold:.3f}): {mid_connectivity:.3f}")
        
        max_connectivity = np.nanmax(dynamic_results.algebraic_connectivity_evolution)
        max_idx = np.nanargmax(dynamic_results.algebraic_connectivity_evolution)
        max_threshold = dynamic_results.thresholds[max_idx]
        
        print(f"Maximum connectivity: {max_connectivity:.3f} at θ={max_threshold:.3f}")