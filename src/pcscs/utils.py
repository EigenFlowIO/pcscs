"""Utility functions for PCSCS analysis."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Optional


def compute_similarity_matrix(features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Compute cosine similarity matrix from feature vectors.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features).
        
    Returns:
        Cosine similarity matrix of shape (n_samples, n_samples).
    """
    if isinstance(features, torch.Tensor):
        # Use PyTorch for GPU acceleration if available
        normalized_features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        return similarity_matrix.cpu().numpy()
    else:
        # Use NumPy for CPU computation
        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized_features = features / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_features, normalized_features.T)
        
        return similarity_matrix


def validate_similarity_matrix(similarity_matrix: np.ndarray) -> None:
    """
    Validate that similarity matrix meets requirements for PCSCS analysis.
    
    Args:
        similarity_matrix: Square similarity matrix to validate.
        
    Raises:
        ValueError: If matrix doesn't meet requirements.
    """
    if not isinstance(similarity_matrix, np.ndarray):
        raise ValueError("Similarity matrix must be numpy array")
    
    if len(similarity_matrix.shape) != 2:
        raise ValueError("Similarity matrix must be 2-dimensional")
    
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Similarity matrix must be square")
    
    if similarity_matrix.shape[0] < 2:
        raise ValueError("Need at least 2 samples for analysis")
    
    # Check diagonal elements are approximately 1 (self-similarity)
    diagonal = np.diag(similarity_matrix)
    if not np.allclose(diagonal, 1.0, atol=1e-6):
        print("Warning: Diagonal elements are not 1.0 (self-similarity should be 1.0)")
    
    # Check matrix is symmetric
    if not np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-6):
        print("Warning: Similarity matrix is not symmetric")
    
    # Check values are in reasonable range for cosine similarity
    if np.any(similarity_matrix < -1.1) or np.any(similarity_matrix > 1.1):
        print("Warning: Similarity values outside expected range [-1, 1]")


def extract_layer_features(activations: torch.Tensor, 
                         flatten: bool = True) -> torch.Tensor:
    """
    Extract and optionally flatten layer activations.
    
    Args:
        activations: Raw layer activations.
        flatten: Whether to flatten spatial dimensions.
        
    Returns:
        Processed feature tensor.
    """
    if flatten and len(activations.shape) > 2:
        # Flatten spatial dimensions, keep batch and channel dims
        return activations.flatten(start_dim=1)
    return activations


def create_threshold_sequence(start: float = 1.0, 
                            end: float = 0.0, 
                            n_steps: int = 1000) -> np.ndarray:
    """
    Create sequence of similarity thresholds for progressive analysis.
    
    Args:
        start: Starting similarity threshold.
        end: Ending similarity threshold.
        n_steps: Number of threshold steps.
        
    Returns:
        Array of threshold values.
    """
    if start <= end:
        raise ValueError("Start threshold must be greater than end threshold")
    
    if n_steps < 2:
        raise ValueError("Need at least 2 steps")
    
    return np.linspace(start, end, n_steps)


def safe_divide(numerator: Union[float, np.ndarray], 
               denominator: Union[float, np.ndarray], 
               default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely divide two values, returning default for division by zero.
    
    Args:
        numerator: Numerator value(s).
        denominator: Denominator value(s).
        default: Value to return when denominator is zero.
        
    Returns:
        Division result with safe handling of zero division.
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(denominator != 0, numerator / denominator, default)
    else:
        result = numerator / denominator if denominator != 0 else default
    
    return result


def format_layer_name(layer_name: str) -> str:
    """
    Format layer name for consistent display.
    
    Args:
        layer_name: Raw layer name.
        
    Returns:
        Formatted layer name.
    """
    # Remove common prefixes and clean up formatting
    formatted = layer_name.replace('features.', 'Layer ')
    formatted = formatted.replace('_', ' ')
    return formatted


def get_default_device() -> str:
    """
    Get default PyTorch device (CUDA if available, otherwise CPU).
    
    Returns:
        Device string ('cuda' or 'cpu').
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def ensure_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Ensure input is numpy array, converting from torch tensor if needed.
    
    Args:
        array: Input array or tensor.
        
    Returns:
        NumPy array.
    """
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def ensure_tensor(array: Union[np.ndarray, torch.Tensor], 
                 device: Optional[str] = None) -> torch.Tensor:
    """
    Ensure input is torch tensor, converting from numpy if needed.
    
    Args:
        array: Input array or tensor.
        device: Target device for tensor.
        
    Returns:
        PyTorch tensor.
    """
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor