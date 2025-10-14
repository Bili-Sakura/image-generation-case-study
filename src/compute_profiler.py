"""
Compute profiler for calculating FLOPs and MACs during inference.
"""

import torch
from typing import Dict, Optional, Tuple
from diffusers import DiffusionPipeline
import time


class ComputeProfiler:
    """Profiler for measuring FLOPs, MACs, and inference time."""
    
    def __init__(self, enabled: bool = True):
        """Initialize the compute profiler.
        
        Args:
            enabled: Whether to enable profiling (can be disabled for performance).
        """
        self.enabled = enabled
        self._calflops_available = False
        
        if self.enabled:
            try:
                from calflops import calculate_flops
                self._calflops_available = True
                self._calculate_flops = calculate_flops
            except ImportError:
                print("⚠️  calflops not installed. Install with: pip install calflops")
                print("   FLOPs/MACs profiling will be disabled.")
                self.enabled = False
    
    def profile_pipeline(
        self,
        pipe: DiffusionPipeline,
        input_shape: Tuple[int, int, int, int],
        num_inference_steps: int = 50,
        model_id: str = "unknown",
    ) -> Dict[str, any]:
        """Profile a diffusion pipeline to calculate FLOPs and MACs.
        
        Args:
            pipe: The diffusion pipeline to profile
            input_shape: Input tensor shape (batch_size, channels, height, width)
            num_inference_steps: Number of inference steps
            model_id: Model identifier for reporting
            
        Returns:
            Dictionary containing FLOPs, MACs, parameters, and other metrics
        """
        if not self.enabled or not self._calflops_available:
            return {
                "enabled": False,
                "total_flops": 0,
                "total_macs": 0,
                "total_params": 0,
                "flops_per_step": 0,
                "macs_per_step": 0,
            }
        
        try:
            # Profile the UNet/Transformer (main compute component)
            model_to_profile = None
            model_name = "model"
            
            # Try to get the main model component
            if hasattr(pipe, 'transformer') and pipe.transformer is not None:
                model_to_profile = pipe.transformer
                model_name = "transformer"
            elif hasattr(pipe, 'unet') and pipe.unet is not None:
                model_to_profile = pipe.unet
                model_name = "unet"
            
            if model_to_profile is None:
                print(f"⚠️  Could not find transformer/unet in pipeline for {model_id}")
                return self._empty_profile()
            
            # Calculate FLOPs for a single forward pass
            # Note: We use a dummy input to estimate compute
            print(f"📊 Profiling {model_name} for {model_id}...")
            
            # Create input args for the model
            batch_size, channels, height, width = input_shape
            latent_height = height // 8  # Common VAE downsampling factor
            latent_width = width // 8
            
            # Prepare kwargs for calflops based on model type
            flops_per_step = 0
            macs_per_step = 0
            params = 0
            
            # Get device - handle accelerate hooks
            try:
                device = next(model_to_profile.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if model_name == "transformer":
                # For transformer-based models (FLUX, SD3, etc.)
                # Most transformers take latent inputs
                latent_channels = 16  # Common for transformers
                input_shape_for_profile = (batch_size, latent_channels, latent_height, latent_width)
                
                # Transformers typically need: hidden_states, timestep, encoder_hidden_states
                try:
                    flops, macs, params = self._calculate_flops(
                        model=model_to_profile,
                        input_shape=input_shape_for_profile,
                        print_results=False,
                        print_detailed=False,
                    )
                    flops_per_step = flops
                    macs_per_step = macs
                except Exception as e:
                    print(f"⚠️  Could not calculate FLOPs for transformer {model_id}: {e}")
                    try:
                        params = sum(p.numel() for p in model_to_profile.parameters())
                    except:
                        pass
            else:
                # For UNet-based models - need sample, timestep, encoder_hidden_states
                latent_channels = 4  # Common for UNet latent space
                input_shape_for_profile = (batch_size, latent_channels, latent_height, latent_width)
                
                try:
                    # Get encoder hidden states dimension from model config
                    if hasattr(model_to_profile, 'config'):
                        encoder_dim = getattr(model_to_profile.config, 'cross_attention_dim', 1024)
                    else:
                        encoder_dim = 1024  # Default
                    
                    # Create wrapper module class
                    class UNetWrapper(torch.nn.Module):
                        """Wrapper module for UNet that handles additional inputs."""
                        def __init__(self, unet, encoder_dim):
                            super().__init__()
                            self.unet = unet
                            self.encoder_dim = encoder_dim
                        
                        def forward(self, sample, timestep=None, encoder_hidden_states=None, **kwargs):
                            """Forward with all required UNet arguments."""
                            if timestep is None:
                                timestep = torch.tensor([1], device=sample.device)
                            if encoder_hidden_states is None:
                                encoder_hidden_states = torch.randn(
                                    sample.shape[0], 77, self.encoder_dim,
                                    device=sample.device, dtype=sample.dtype
                                )
                            return self.unet(
                                sample=sample,
                                timestep=timestep,
                                encoder_hidden_states=encoder_hidden_states,
                                return_dict=False,
                                **kwargs
                            )
                    
                    # Create wrapper instance
                    wrapped_model = UNetWrapper(model_to_profile, encoder_dim)
                    
                    # Capture calflops output - it may return strings or numbers depending on version
                    result = self._calculate_flops(
                        model=wrapped_model,
                        input_shape=input_shape_for_profile,
                        print_results=False,
                        print_detailed=False,
                    )
                    
                    # Handle different return formats from calflops
                    if isinstance(result, tuple) and len(result) >= 3:
                        flops_val, macs_val, params_val = result[0], result[1], result[2]
                        
                        # Convert to float if they're strings
                        if isinstance(flops_val, str):
                            # Extract numeric value from formatted string like "678.72 GFLOPS"
                            try:
                                # Remove repeated strings if present
                                if "FLOPS" in flops_val:
                                    flops_val = flops_val.split("FLOPS")[0].strip()
                                flops_per_step = self._parse_formatted_value(flops_val)
                            except:
                                flops_per_step = 0
                        else:
                            flops_per_step = float(flops_val) if flops_val else 0
                        
                        if isinstance(macs_val, str):
                            try:
                                if "MACS" in macs_val or "MACs" in macs_val:
                                    macs_val = macs_val.split("MAC")[0].strip()
                                macs_per_step = self._parse_formatted_value(macs_val)
                            except:
                                macs_per_step = 0
                        else:
                            macs_per_step = float(macs_val) if macs_val else 0
                        
                        if isinstance(params_val, str):
                            try:
                                params = self._parse_formatted_value(params_val)
                            except:
                                params = sum(p.numel() for p in model_to_profile.parameters())
                        else:
                            params = int(params_val) if params_val else 0
                    else:
                        flops_per_step = 0
                        macs_per_step = 0
                        
                except Exception as e:
                    print(f"⚠️  Could not calculate FLOPs for {model_id}: {e}")
                    print(f"   This may be due to model using accelerate hooks or unsupported operations.")
                    # Try to at least count parameters
                    try:
                        params = sum(p.numel() for p in model_to_profile.parameters())
                    except:
                        pass
            
            # Multiply by number of inference steps to get total compute
            total_flops = flops_per_step * num_inference_steps if flops_per_step else 0
            total_macs = macs_per_step * num_inference_steps if macs_per_step else 0
            
            return {
                "enabled": True,
                "model_component": model_name,
                "total_flops": total_flops,
                "total_macs": total_macs,
                "total_params": params,
                "flops_per_step": flops_per_step,
                "macs_per_step": macs_per_step,
                "num_inference_steps": num_inference_steps,
                "input_shape": input_shape,
                "latent_shape": input_shape_for_profile,
                # Human readable formats
                "total_flops_str": self._format_compute(total_flops, "FLOPs"),
                "total_macs_str": self._format_compute(total_macs, "MACs"),
                "params_str": self._format_params(params),
                "flops_per_step_str": self._format_compute(flops_per_step, "FLOPs"),
                "macs_per_step_str": self._format_compute(macs_per_step, "MACs"),
            }
            
        except Exception as e:
            print(f"⚠️  Error during profiling: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_profile()
    
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
        """Format compute value in human-readable form.
        
        Args:
            value: Raw compute value
            unit: Unit name (FLOPs or MACs)
            
        Returns:
            Human-readable string
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
    
    def _format_params(self, value: int) -> str:
        """Format parameter count in human-readable form.
        
        Args:
            value: Number of parameters
            
        Returns:
            Human-readable string
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
    
    def _parse_formatted_value(self, value_str: str) -> float:
        """Parse a formatted value string back to numeric value.
        
        Args:
            value_str: Formatted string like "678.72" or "678.72 G"
            
        Returns:
            Numeric value
        """
        if not value_str or not isinstance(value_str, str):
            return 0.0
        
        # Remove whitespace
        value_str = value_str.strip()
        
        # Split number and unit
        parts = value_str.split()
        if not parts:
            return 0.0
        
        try:
            number = float(parts[0])
        except ValueError:
            return 0.0
        
        # Check for unit multiplier
        if len(parts) > 1:
            unit = parts[1].upper()
            if unit.startswith('T'):
                number *= 1e12
            elif unit.startswith('G'):
                number *= 1e9
            elif unit.startswith('M'):
                number *= 1e6
            elif unit.startswith('K'):
                number *= 1e3
        
        return number


def create_profiler(enabled: bool = True) -> ComputeProfiler:
    """Factory function to create a compute profiler.
    
    Args:
        enabled: Whether to enable profiling
        
    Returns:
        ComputeProfiler instance
    """
    return ComputeProfiler(enabled=enabled)

