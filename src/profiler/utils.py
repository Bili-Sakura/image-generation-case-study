"""
Utility functions for compute profiling.
Clean version using calflops (no THOP or manual SDPA tracking needed).
"""

import torch
from typing import Optional


def format_compute(value: float, unit: str = "FLOPs") -> str:
    """
    Format compute value in human-readable form.
    
    Args:
        value: Raw compute value
        unit: Unit name (FLOPs or MACs)
    
    Returns:
        Human-readable string (e.g., "12.34 GFLOPs")
    """
    if value == 0:
        return f"0 {unit}"
    
    # Use SI prefixes
    prefixes = ["", "K", "M", "G", "T", "P", "E"]
    magnitude = 0
    scaled_value = float(value)
    
    while scaled_value >= 1000.0 and magnitude < len(prefixes) - 1:
        scaled_value /= 1000.0
        magnitude += 1
    
    return f"{scaled_value:.2f} {prefixes[magnitude]}{unit}"


def format_params(value: int) -> str:
    """
    Format parameter count in human-readable form.
    
    Args:
        value: Number of parameters
    
    Returns:
        Human-readable string (e.g., "1.23B")
    """
    if value == 0:
        return "0"
    
    # Use SI prefixes
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return str(value)


def to_gmacs(macs: int) -> float:
    """Convert MACs to GigaMACs."""
    return macs / 1e9


def to_gflops(flops: int) -> float:
    """Convert FLOPs to GigaFLOPs."""
    return flops / 1e9


def get_model_device(model) -> torch.device:
    """Get the device of a model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_model_dtype(model) -> torch.dtype:
    """Get the dtype of a model."""
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def count_parameters(model) -> int:
    """Count total parameters in a model."""
    try:
        return sum(p.numel() for p in model.parameters())
    except:
        return 0


def detect_latent_channels(pipe) -> int:
    """
    Detect latent channels from pipeline VAE config.
    
    Args:
        pipe: Diffusion pipeline
    
    Returns:
        Number of latent channels (4 for SD, 16 for FLUX/SD3)
    """
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        if hasattr(pipe.vae, 'config'):
            return getattr(pipe.vae.config, 'latent_channels', 4)
    return 4


def detect_vae_scaling_factor(pipe) -> float:
    """
    Detect VAE scaling factor from pipeline config.
    
    Args:
        pipe: Diffusion pipeline
    
    Returns:
        VAE scaling factor
    """
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        if hasattr(pipe.vae, 'config'):
            return getattr(pipe.vae.config, 'scaling_factor', 0.18215)
    return 0.18215


def get_embedding_dim(model, default: int = 4096) -> int:
    """
    Get embedding dimension from model config.
    Tries multiple config attributes commonly used for embedding dims.
    
    Args:
        model: The model to inspect
        default: Default value if not found
    
    Returns:
        Embedding dimension
    """
    if not hasattr(model, 'config'):
        return default
    
    # Try different attribute names
    candidates = [
        'joint_attention_dim',
        'caption_channels',
        'cap_feat_dim',
        'cross_attention_dim',
        'hidden_size',
    ]
    
    for attr in candidates:
        value = getattr(model.config, attr, None)
        if value is not None:
            return value
    
    return default


def get_pooled_projection_dim(model, default: int = 768) -> int:
    """
    Get pooled projection dimension from model config.
    
    Args:
        model: The model to inspect
        default: Default value if not found
    
    Returns:
        Pooled projection dimension
    """
    if hasattr(model, 'config'):
        return getattr(model.config, 'pooled_projection_dim', default)
    return default
