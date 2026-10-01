"""Mathematical analysis functions for PCSCS."""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional


def sigmoid_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Sigmoid function: f(x) = a / (1 + exp(-b*(x-c))) + d
    
    Args:
        x: Input values.
        a: Amplitude parameter.
        b: Steepness parameter.
        c: Midpoint parameter.
        d: Offset parameter.
        
    Returns:
        Sigmoid function values.
    """
    return a / (1 + np.exp(-b * (x - c))) + d


def fit_sigmoid(thresholds: np.ndarray, num_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    """
    Fit sigmoid function to threshold vs num_classes data.
    
    Args:
        thresholds: Array of similarity thresholds (decreasing).
        num_classes: Array of number of classes at each threshold.
        
    Returns:
        Tuple of (smooth_thresholds, smooth_classes, sigmoid_params, fit_success).
    """
    # Transform data: work from low to high similarity thresholds
    x_data = thresholds[::-1]  # Now increasing from convergence to 1.0
    y_data = num_classes[::-1]  # Corresponding y values
    
    # Initial parameter guesses
    a_init = np.max(y_data) - np.min(y_data)  # amplitude
    b_init = 10.0  # steepness
    c_init = np.mean(x_data)  # midpoint
    d_init = np.min(y_data)  # offset
    
    try:
        # Fit sigmoid
        popt, _ = curve_fit(
            sigmoid_function,
            x_data,
            y_data,
            p0=[a_init, b_init, c_init, d_init],
            maxfev=10000
        )
        
        # Generate smooth sigmoid curve
        x_smooth = np.linspace(x_data[0], x_data[-1], 1000)
        y_smooth = sigmoid_function(x_smooth, *popt)
        
        # Convert back to original order (decreasing thresholds)
        x_smooth_orig = x_smooth[::-1]
        y_smooth_orig = y_smooth[::-1]
        
        return x_smooth_orig, y_smooth_orig, popt, True
        
    except Exception:
        # Fallback to linear interpolation
        x_smooth = np.linspace(thresholds[0], thresholds[-1], 1000)
        y_smooth = np.interp(x_smooth[::-1], thresholds[::-1], num_classes[::-1])[::-1]
        return x_smooth, y_smooth, None, False


def find_critical_threshold(smooth_thresholds: np.ndarray, 
                          smooth_classes: np.ndarray,
                          sigmoid_params: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray]:
    """
    Find similarity threshold with maximum rate of class increase.
    
    Args:
        smooth_thresholds: Smooth threshold values.
        smooth_classes: Smooth class count values.
        sigmoid_params: Optional sigmoid parameters for analytical derivative.
        
    Returns:
        Tuple of (critical_threshold, critical_rate, derivative_array).
    """
    # Work with increasing x values (reverse arrays)
    x_increasing = smooth_thresholds[::-1]
    y_increasing = smooth_classes[::-1]
    
    if sigmoid_params is not None:
        # Analytical derivative of sigmoid
        a, b, c, d = sigmoid_params
        
        def sigmoid_derivative(x):
            exp_term = np.exp(-b * (x - c))
            return (a * b * exp_term) / ((1 + exp_term) ** 2)
        
        derivative = sigmoid_derivative(x_increasing)
    else:
        # Numerical derivative
        derivative = np.gradient(y_increasing, x_increasing)
    
    # Find maximum increase rate
    max_increase_idx = np.argmax(derivative)
    max_increase_threshold = x_increasing[max_increase_idx]
    max_increase_rate = derivative[max_increase_idx]
    
    # Convert derivative back to original order for plotting
    derivative_orig = derivative[::-1]
    
    return max_increase_threshold, max_increase_rate, derivative_orig