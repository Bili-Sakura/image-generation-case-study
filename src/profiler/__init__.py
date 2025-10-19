"""
Simplified compute profiler for diffusion models.

Philosophy: Read from model files first, config.yaml as fallback only.

This package provides:
- Automatic model discovery from model directories
- Architecture inference from model JSON files
- Component-level profiling (main model, VAE, text encoders)
- Support for all major architectures (UNet, Transformer-based models)
"""

from .profiler import ComputeProfiler, create_profiler
from .model_reader import ModelReader
from .architecture_matcher import ArchitectureMatcher
from .utils import format_compute, format_params, to_gmacs, to_gflops

__all__ = [
    'ComputeProfiler',
    'create_profiler',
    'ModelReader',
    'ArchitectureMatcher',
    'format_compute',
    'format_params',
    'to_gmacs',
    'to_gflops',
]

__version__ = '2.0.0'

