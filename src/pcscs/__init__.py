"""
Progressive Cosine Similarity Classification Sifting (PCSCS)

A comprehensive framework for analyzing neural network representations through 
progressive similarity-based clustering with hierarchical tree construction 
and critical threshold detection.
"""

from .core import PCSCS
from .analysis import fit_sigmoid, find_critical_threshold, sigmoid_function
from .utils import compute_similarity_matrix

__version__ = "0.1.0"
__author__ = "Andrew Hedman"

__all__ = [
    "PCSCS",
    "fit_sigmoid", 
    "find_critical_threshold",
    "sigmoid_function",
    "compute_similarity_matrix"
]