"""
Backward-compatible wrapper for the new modular compute profiler.
This file maintains the old API while using the new profiler system internally.

The new modular profiler is located in src/profiler/ directory.
For new code, prefer importing directly from src.profiler instead.
"""

from typing import Dict, Optional, Tuple, Any
from diffusers import DiffusionPipeline

# Import new modular profiler
from src.profiler import create_profiler as create_new_profiler


class ComputeProfiler:
    """
    Backward-compatible compute profiler wrapper.
    Delegates to the new modular profiler while maintaining the old API.
    
    For new code, use: from profiler import create_profiler
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the compute profiler.
        
        Args:
            enabled: Whether to enable profiling (can be disabled for performance).
        """
        self.enabled = enabled
        self._profiler = create_new_profiler(enabled=enabled)
        # Backward compatibility: legacy code expects `_thop_available`; the new profiler
        # uses `_calflops_available`. Support both and default to False if missing.
        self._thop_available = getattr(self._profiler, "_thop_available", False) or getattr(self._profiler, "_calflops_available", False)
    
    def get_model_config(self, model_id: str) -> Optional[Dict]:
        """
        Get model configuration.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model configuration dictionary or None if not found
        """
        registry = self._profiler.detector.model_registry
        return registry.get_model_config(model_id)
    
    def detect_model_components(self, pipe, model_id: Optional[str] = None) -> Dict[str, any]:
        """
        Detect what components a model has (legacy API).
        
        Args:
            pipe: The diffusion pipeline
            model_id: Optional model identifier
        
        Returns:
            Dictionary with component information
        """
        return self._profiler.detector.detect_components(pipe)
    
    def measure_unet_macs(
        self,
        pipe,
        height: int = 512,
        width: int = 512,
        prompt: str = "a photo of a cat",
        guidance_scale: float = 7.5,
        model_id: Optional[str] = None
    ) -> int:
        """
        Measure MACs for UNet/Transformer forward pass (legacy API).
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
            prompt: Text prompt for conditioning
            guidance_scale: Guidance scale
            model_id: Optional model identifier
        
        Returns:
            MACs per inference step
        """
        if not self.enabled or not self._thop_available:
            return 0
        
        # Detect architecture
        arch_info = self._profiler.detect_architecture(pipe, model_id)
        if arch_info is None:
            return 0
        
        # Profile main model
        macs, _ = self._profiler.profile_main_model(
            pipe, arch_info, height, width, guidance_scale, verbose=False
        )
        return macs
    
    def measure_vae_decode_macs(self, pipe, height: int = 512, width: int = 512) -> int:
        """
        Measure MACs for VAE decoder (legacy API).
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
        
        Returns:
            MACs for VAE decode operation
        """
        if not self.enabled or not self._thop_available:
            return 0
        
        # Detect architecture
        arch_info = self._profiler.detect_architecture(pipe)
        if arch_info is None:
            return 0
        
        # Profile VAE
        macs, _ = self._profiler.profile_vae(
            pipe, arch_info, height, width, verbose=False
        )
        return macs
    
    def measure_text_encoder_macs(
        self,
        pipe,
        prompt: str = "a photo of a cat",
        negative_prompt: str = "",
        model_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Measure MACs for text encoder(s) (legacy API).
        
        Args:
            pipe: The diffusion pipeline
            prompt: Text prompt
            negative_prompt: Negative text prompt
            model_id: Optional model identifier
        
        Returns:
            Dictionary with MACs for each text encoder
        """
        if not self.enabled or not self._thop_available:
            return {"total": 0, "text_encoder_1": 0, "text_encoder_2": 0, "text_encoder_3": 0, "text_encoder_4": 0}
        
        # Detect architecture
        arch_info = self._profiler.detect_architecture(pipe, model_id)
        if arch_info is None:
            return {"total": 0, "text_encoder_1": 0, "text_encoder_2": 0, "text_encoder_3": 0, "text_encoder_4": 0}
        
        # Profile text encoders
        return self._profiler.profile_text_encoders(
            pipe, arch_info, prompt, negative_prompt, verbose=False
        )
    
    def summarize_macs(
        self,
        pipe,
        height: int = 512,
        width: int = 512,
        steps: int = 30,
        prompt: str = "a photo of a cat",
        guidance_scale: float = 7.5,
        model_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Summarize MACs for complete image generation (legacy API).
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
            steps: Number of inference steps
            prompt: Text prompt
            guidance_scale: Guidance scale
            model_id: Optional model identifier
        
        Returns:
            Dictionary with MAC counts and formatted strings
        """
        return self._profiler.summarize(
            pipe, height, width, steps, prompt, guidance_scale, model_id, verbose=False
        )
    
    def profile_pipeline(
        self,
        pipe: DiffusionPipeline,
        input_shape: Tuple[int, int, int, int],
        num_inference_steps: int = 50,
        model_id: str = "unknown",
        guidance_scale: float = 7.5,
    ) -> Dict[str, any]:
        """
        Profile a diffusion pipeline to calculate FLOPs and MACs (legacy API).
        
        Args:
            pipe: The diffusion pipeline to profile
            input_shape: Input tensor shape (batch_size, channels, height, width)
            num_inference_steps: Number of inference steps
            model_id: Model identifier for reporting
            guidance_scale: Guidance scale for generation
        
        Returns:
            Dictionary containing FLOPs, MACs, parameters, and other metrics
        """
        batch_size, channels, height, width = input_shape
        
        summary = self._profiler.summarize(
            pipe, height, width, num_inference_steps, "a photo", guidance_scale, model_id
        )
        
        if not summary.get("enabled", False):
            return self._empty_profile()
        
        # Extract values for legacy format
        total_macs = summary.get(f"total_{num_inference_steps}_steps_macs", 0)
        total_flops = total_macs * 2  # FLOPs ≈ 2 × MACs
        macs_per_step = summary.get("main_model_per_step_macs", 0)
        flops_per_step = macs_per_step * 2
        params = summary.get("total_params", 0)
        
        return {
            "enabled": True,
            "model_component": summary.get("architecture", "unknown"),
            "total_flops": total_flops,
            "total_macs": total_macs,
            "total_params": params,
            "flops_per_step": flops_per_step,
            "macs_per_step": macs_per_step,
            "num_inference_steps": num_inference_steps,
            "input_shape": input_shape,
            # Human readable formats
            "total_flops_str": self._format_compute(total_flops, "FLOPs"),
            "total_macs_str": self._format_compute(total_macs, "MACs"),
            "params_str": self._format_params(params),
            "flops_per_step_str": self._format_compute(flops_per_step, "FLOPs"),
            "macs_per_step_str": self._format_compute(macs_per_step, "MACs"),
            # Additional breakdowns
            "unet_macs_per_step": macs_per_step,
            "unet_macs_per_step_str": self._format_compute(macs_per_step, "MACs"),
            "vae_macs": summary.get("vae_macs", 0),
            "vae_macs_str": self._format_compute(summary.get("vae_macs", 0), "MACs"),
            "text_encoder_macs": summary.get("text_encoder_total_macs", 0),
            "text_encoder_macs_str": self._format_compute(summary.get("text_encoder_total_macs", 0), "MACs"),
        }
    
    def _empty_profile(self) -> Dict[str, any]:
        """Return an empty profile when profiling fails."""
        return {
            "enabled": False,
            "total_flops": 0,
            "total_macs": 0,
            "total_params": 0,
            "flops_per_step": 0,
            "macs_per_step": 0,
        }
    
    def _format_compute(self, value: float, unit: str = "FLOPs") -> str:
        """Format compute value in human-readable form."""
        from src.profiler.utils import format_compute
        return format_compute(value, unit)
    
    def _format_params(self, value: int) -> str:
        """Format parameter count in human-readable form."""
        from src.profiler.utils import format_params
        return format_params(value)


def create_profiler(enabled: bool = True) -> ComputeProfiler:
    """
    Factory function to create a compute profiler (legacy API).
    
    For new code, prefer: from profiler import create_profiler
    
    Args:
        enabled: Whether to enable profiling
    
    Returns:
        ComputeProfiler instance
    """
    return ComputeProfiler(enabled=enabled)

