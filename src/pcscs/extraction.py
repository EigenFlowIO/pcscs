"""PyTorch model feature extraction with hooks (domain-agnostic)."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from .utils import compute_similarity_matrix


class FeatureExtractor:
    """Extract features from PyTorch models using forward hooks."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """
        Initialize feature extractor.
        
        Args:
            model: PyTorch model to extract features from.
            device: Device to run computations on.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.activations = {}
        self.hooks = []
    
    def _get_conv_layer_names(self) -> List[str]:
        """Get names of all convolutional layers."""
        layer_names = []
        
        if hasattr(self.model, 'features'):
            for i, layer in enumerate(self.model.features):
                if isinstance(layer, torch.nn.Conv2d):
                    layer_names.append(f"features.{i}")
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    layer_names.append(name)
                    
        return layer_names
    
    def _get_layer_by_name(self, name: str) -> torch.nn.Module:
        """Get layer by name string."""
        parts = name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def _register_hooks(self, layer_names: List[str]) -> None:
        """Register forward hooks for specified layers."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Clear existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Register new hooks
        for name in layer_names:
            try:
                layer = self._get_layer_by_name(name)
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
            except AttributeError:
                raise ValueError(f"Layer {name} not found in model")
    
    def extract_features(self, samples: List[Any], 
                        preprocess_fn: Callable,
                        layer_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Extract features from model layers.
        
        Args:
            samples: List of samples (any type that preprocess_fn can handle).
            preprocess_fn: Function to preprocess samples into tensors.
            layer_names: Layer names to extract from. If None, uses all conv layers.
            
        Returns:
            Dictionary mapping layer names to feature data including similarity matrices.
        """
        if layer_names is None:
            layer_names = self._get_conv_layer_names()
        
        self._register_hooks(layer_names)
        
        # Process samples and extract features
        all_layer_features = {name: [] for name in layer_names}
        
        for sample in samples:
            # Preprocess sample (user's responsibility to convert to tensor)
            input_tensor = preprocess_fn(sample)
            
            # Ensure proper tensor dimensions
            if input_tensor.dim() == 3:  # Add batch dimension if needed
                input_tensor = input_tensor.unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device)
            
            # Forward pass to trigger hooks
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            # Store activations for each layer
            for layer_name in layer_names:
                if layer_name in self.activations:
                    # Flatten spatial dimensions
                    features = self.activations[layer_name].flatten(start_dim=1)
                    all_layer_features[layer_name].append(features.squeeze(0))
        
        # Compute similarity matrices for each layer
        layer_results = {}
        for layer_name in layer_names:
            if all_layer_features[layer_name]:
                # Stack all features for this layer
                layer_features = torch.stack(all_layer_features[layer_name])
                
                # Compute similarity matrix
                similarity_matrix = compute_similarity_matrix(layer_features.cpu().numpy())
                
                layer_results[layer_name] = {
                    'features': layer_features.cpu().numpy(),
                    'similarity_matrix': similarity_matrix,
                    'feature_shape': layer_features.shape,
                    'n_samples': len(samples)
                }
        
        # Clean up hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return layer_results


def extract_features_from_model(samples: List[Any],
                               model: torch.nn.Module,
                               preprocess_fn: Callable,
                               layer_names: Optional[List[str]] = None,
                               device: str = 'cuda') -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to extract features from PyTorch model.
    
    Args:
        samples: List of samples to process.
        model: PyTorch model.
        preprocess_fn: Sample preprocessing function.
        layer_names: Specific layers to extract from.
        device: Device for computation.
        
    Returns:
        Dictionary of layer features and similarity matrices.
    """
    extractor = FeatureExtractor(model, device)
    return extractor.extract_features(samples, preprocess_fn, layer_names)