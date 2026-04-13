import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class DPEngine:
    """Provides Differential Privacy and Efficiency mechanisms for FL updates."""
    
    @staticmethod
    def apply_dp(weights: Dict[str, torch.Tensor], clip_threshold: float, noise_multiplier: float) -> Dict[str, torch.Tensor]:
        """Applies clipping and Gaussian noise to the weights."""
        if clip_threshold <= 0:
            return weights
            
        processed_weights = {}
        # 1. Calculate global norm for clipping
        total_norm = torch.norm(torch.stack([torch.norm(v.float()) for v in weights.values()]))
        
        # 2. Clipping factor
        clip_factor = min(1.0, clip_threshold / (total_norm + 1e-6))
        
        for k, v in weights.items():
            # Apply Clipping
            v_clipped = v * clip_factor
            
            # 3. Add Gaussian Noise
            if noise_multiplier > 0:
                std = noise_multiplier * clip_threshold
                noise = torch.randn_like(v_clipped.float()) * std
                processed_weights[k] = v_clipped + noise
            else:
                processed_weights[k] = v_clipped
                
        return processed_weights

    @staticmethod
    def apply_quantization(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Simple FP16 quantization to reduce bandwidth by 50%."""
        return {k: v.half() for k, v in weights.items()}

    @staticmethod
    def apply_sparsification(weights: Dict[str, torch.Tensor], top_k_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """Keeps only top-k percentage of gradients (Sparsification)."""
        if top_k_ratio >= 1.0:
            return weights
            
        processed_weights = {}
        for k, v in weights.items():
            flat_v = v.flatten()
            num_keep = max(1, int(len(flat_v) * top_k_ratio))
            
            # Get values and indices of top-k absolute values
            values, indices = torch.topk(torch.abs(flat_v), num_keep)
            
            # Create a sparse-like mask (keeping only top values)
            mask = torch.zeros_like(flat_v)
            mask[indices] = flat_v[indices]
            processed_weights[k] = mask.reshape(v.shape)
            
        return processed_weights
