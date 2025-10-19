"""
Component-level profilers for different parts of diffusion models.
Uses calflops for accurate FLOPs/MACs calculation with automatic attention handling.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from .utils import (
    get_model_device,
    get_model_dtype,
    detect_vae_scaling_factor,
    count_parameters,
)
from .wrappers import get_wrapper_class, VAEWrapper, TextEncoderWrapper


def extract_model_inputs_from_pipeline(pipeline_inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract and prepare real inputs from pipeline inference for profiling.
    
    This function takes actual inference inputs captured from the pipeline
    and prepares them for profiling the main model (transformer/unet).
    
    Args:
        pipeline_inputs: Dictionary containing real inputs from pipeline inference
            Expected keys may include:
            - 'sample': Latent tensor
            - 'timestep': Timestep tensor
            - 'encoder_hidden_states': Text embeddings
            - 'pooled_projections': Pooled text embeddings (if applicable)
            - 'img_ids': Image position IDs (FLUX)
            - 'txt_ids': Text position IDs (FLUX)
            - 'guidance': Guidance scale tensor
            - 'encoder_mask': Encoder attention mask
            - 'encoder_attention_mask': Attention mask
            - 'image_rotary_emb': Rotary embeddings
            - 'original_size': Original image size
            - 'target_size': Target image size
            - 'crop_coords': Crop coordinates
            - 'added_cond_kwargs': Additional conditioning (SDXL)
    
    Returns:
        Dictionary of input tensors ready for profiling
    """
    if not pipeline_inputs:
        raise ValueError("pipeline_inputs cannot be empty. Must provide real inputs from pipeline inference.")
    
    # Verify required base inputs
    required_keys = ['sample', 'timestep', 'encoder_hidden_states']
    missing_keys = [key for key in required_keys if key not in pipeline_inputs]
    if missing_keys:
        raise ValueError(f"Missing required inputs: {missing_keys}. These must be captured from pipeline inference.")
    
    # Return the inputs as-is since they're already from real inference
    return pipeline_inputs


class MainModelProfiler:
    """Profiler for main model (UNet/Transformer) using calflops."""
    
    def __init__(self, calflops_module):
        """
        Initialize main model profiler.
        
        Args:
            calflops_module: The calflops module
        """
        self.calflops = calflops_module
    
    def profile(
        self,
        model,
        arch_config: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
        guidance_scale: float = 7.5,
        verbose: bool = False,
    ) -> Tuple[int, int]:
        """
        Profile main model (UNet/Transformer) using real pipeline inputs.
        
        Args:
            model: The main model to profile
            arch_config: Architecture configuration
            pipeline_inputs: Real inputs captured from pipeline inference
            guidance_scale: Guidance scale (affects CFG multiplier)
            verbose: Whether to print verbose output
        
        Returns:
            Tuple of (macs_per_step, params)
        """
        profiling_config = arch_config.get('profiling')
        if profiling_config is None:
            raise ValueError("arch_config must contain 'profiling' configuration")
        
        # Get wrapper class (required, no default)
        wrapper_name = profiling_config.get('wrapper_class')
        if wrapper_name is None:
            raise ValueError("profiling config must contain 'wrapper_class'")
        
        try:
            WrapperClass = get_wrapper_class(wrapper_name)
        except KeyError as e:
            raise ValueError(f"Invalid wrapper class: {e}")
        
        # Extract real inputs from pipeline
        inputs = extract_model_inputs_from_pipeline(pipeline_inputs)
        
        try:
            # Debug: Check input types
            if verbose:
                for key, val in inputs.items():
                    print(f"  Input '{key}': type={type(val)}, is_tensor={isinstance(val, torch.Tensor)}")
            
            # Create wrapper
            wrapper = self._create_wrapper(WrapperClass, model, inputs)
            wrapper.eval()
            
            # Get sample tensor
            sample = inputs['sample']
            if not isinstance(sample, torch.Tensor):
                raise ValueError(f"Sample must be a tensor, got {type(sample)}")

            # Profile using calflops
            with torch.no_grad():
                flops, macs, params = self.calflops.calculate_flops(
                    model=wrapper,
                    input_shape=tuple(sample.shape),  # Convert torch.Size to tuple
                    output_as_string=False,
                    output_precision=4,
                    print_results=verbose,
                    print_detailed=False,
                )
            
            if verbose:
                print(f"  FLOPs: {flops:,}, MACs: {macs:,}, Params: {params:,}")
        
        except Exception as e:
            if verbose:
                print(f"⚠️  Error profiling main model: {e}")
                import traceback
                traceback.print_exc()
            macs, params = 0, 0
        
        # Apply CFG multiplier (required, no default)
        cfg_multiplier = profiling_config.get('cfg_multiplier')
        if cfg_multiplier is None:
            raise ValueError("profiling config must contain 'cfg_multiplier'")
        
        # Only apply CFG if guidance_scale > 1.0
        if guidance_scale and guidance_scale > 1.0:
            macs = macs * cfg_multiplier
        
        return int(macs), int(params)
    
    def _create_wrapper(self, WrapperClass, model, inputs: Dict[str, Any]):
        """Create appropriate wrapper instance based on inputs."""
        wrapper_name = WrapperClass.__name__
        device = get_model_device(model)
        dtype = get_model_dtype(model)
        
        # Most wrappers can be initialized with the model and the inputs dictionary
        try:
            return WrapperClass(model, **inputs).to(device=device, dtype=dtype)
        except TypeError:
            # Fallback for wrappers with specific signatures
            if wrapper_name == 'UNetWrapper':
                return WrapperClass(
                    model,
                    inputs['timestep'],
                    inputs['encoder_hidden_states'],
                    inputs.get('added_cond_kwargs')
                ).to(device=device, dtype=dtype)
            
            elif wrapper_name == 'FluxTransformerWrapper':
                return WrapperClass(
                    model,
                    inputs['encoder_hidden_states'],
                    inputs['pooled_projections'],
                    inputs['timestep'],
                    inputs['img_ids'],
                    inputs['txt_ids'],
                    inputs.get('guidance')
                ).to(device=device, dtype=dtype)
            
            else:
                # Re-raise if no specific handler is found
                raise


class VAEProfiler:
    """Profiler for VAE decoder using calflops."""
    
    def __init__(self, calflops_module):
        """
        Initialize VAE profiler.
        
        Args:
            calflops_module: The calflops module
        """
        self.calflops = calflops_module
    
    def profile(
        self,
        vae,
        latents: torch.Tensor,
        pipe,
        arch_config: Dict[str, Any],
        verbose: bool = False,
    ) -> Tuple[int, int]:
        """
        Profile VAE decoder using real latents from pipeline inference.
        
        Args:
            vae: VAE model
            latents: Real latent tensors from pipeline inference
            pipe: Pipeline (for config access)
            arch_config: Architecture configuration
            verbose: Whether to print verbose output
        
        Returns:
            Tuple of (macs, params)
        """
        try:
            if latents is None:
                raise ValueError("latents cannot be None. Must provide real latents from pipeline inference.")
            
            vae.eval()
            device = get_model_device(vae)
            dtype = get_model_dtype(vae)
            
            # Use real latents from pipeline
            latents = latents.to(device=device, dtype=dtype)
            
            # Get scaling factor
            scaling_factor = detect_vae_scaling_factor(pipe)
            
            # Create wrapper
            wrapper = VAEWrapper(vae, scaling_factor).to(device=device, dtype=dtype)
            wrapper.eval()
            
            # Profile using calflops
            with torch.no_grad():
                flops, macs, params = self.calflops.calculate_flops(
                    model=wrapper,
                    input_shape=tuple(latents.shape),  # Convert torch.Size to tuple
                    output_as_string=False,
                    output_precision=4,
                    print_results=verbose,
                    print_detailed=False,
                )
            
            if verbose:
                print(f"  VAE - FLOPs: {flops:,}, MACs: {macs:,}, Params: {params:,}")
            
            return int(macs), int(params)
        
        except Exception as e:
            if verbose:
                print(f"⚠️  Error profiling VAE: {e}")
                import traceback
                traceback.print_exc()
            return 0, 0


class TextEncoderProfiler:
    """Profiler for text encoders using calflops."""
    
    def __init__(self, calflops_module):
        """
        Initialize text encoder profiler.
        
        Args:
            calflops_module: The calflops module
        """
        self.calflops = calflops_module
    
    def profile(
        self,
        text_encoder,
        tokenized_inputs: Dict[str, torch.Tensor],
        arch_config: Dict[str, Any],
        verbose: bool = False,
    ) -> Tuple[int, int, int]:
        """
        Profile a single text encoder using real tokenized inputs.
        
        Args:
            text_encoder: Text encoder model
            tokenized_inputs: Real tokenized inputs from pipeline with keys:
                - 'positive': tokenized positive prompt
                - 'negative': tokenized negative prompt
            arch_config: Architecture configuration
            verbose: Whether to print verbose output
        
        Returns:
            Tuple of (total_macs, params, max_length) for both positive and negative prompts
        """
        try:
            if not tokenized_inputs:
                raise ValueError("tokenized_inputs cannot be empty. Must provide real tokenized inputs from pipeline.")
            
            if 'positive' not in tokenized_inputs or 'negative' not in tokenized_inputs:
                raise ValueError("tokenized_inputs must contain 'positive' and 'negative' keys")
            
            text_encoder.eval()
            device = get_model_device(text_encoder)
            dtype = get_model_dtype(text_encoder)
            
            # Get text sequence length from config (required, no default)
            profiling_config = arch_config.get('profiling')
            if profiling_config is None:
                raise ValueError("arch_config must contain 'profiling' configuration")
            
            max_length = profiling_config.get('text_seq_len')
            if max_length is None:
                raise ValueError("profiling config must contain 'text_seq_len'")
            
            # Use real tokenized inputs from pipeline
            with torch.no_grad():
                enc = tokenized_inputs['positive'].to(device)
                neg = tokenized_inputs['negative'].to(device)
                
                # Create wrapper
                wrapper = TextEncoderWrapper(text_encoder).to(device=device, dtype=dtype)
                wrapper.eval()
                
                # Profile positive prompt using calflops
                flops_pos, macs_pos, params = self.calflops.calculate_flops(
                    model=wrapper,
                    input_shape=tuple(enc.input_ids.shape),  # Convert torch.Size to tuple
                    output_as_string=False,
                    output_precision=4,
                    print_results=False,
                    print_detailed=False,
                )
                
                # Profile negative prompt using calflops
                flops_neg, macs_neg, _ = self.calflops.calculate_flops(
                    model=wrapper,
                    input_shape=tuple(neg.input_ids.shape),  # Convert torch.Size to tuple
                    output_as_string=False,
                    output_precision=4,
                    print_results=False,
                    print_detailed=False,
                )
                
                total_macs = macs_pos + macs_neg
                
                if verbose:
                    print(f"  Text Encoder - Total MACs: {total_macs:,}, Params: {params:,}")
                
                return int(total_macs), int(params), int(max_length)
        
        except Exception as e:
            if verbose:
                print(f"⚠️  Error profiling text encoder: {e}")
                import traceback
                traceback.print_exc()
            return 0, 0, 0
    
    def profile_all(
        self,
        pipe,
        tokenized_inputs_all: Dict[str, Dict[str, torch.Tensor]],
        arch_config: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, int]:
        """
        Profile all text encoders in a pipeline using real tokenized inputs.
        
        Args:
            pipe: Pipeline with text encoders
            tokenized_inputs_all: Dictionary mapping encoder names to their tokenized inputs
                Expected keys: 'text_encoder_1', 'text_encoder_2', etc.
                Each value should be a dict with 'positive' and 'negative' keys
            arch_config: Architecture configuration
            verbose: Whether to print verbose output
        
        Returns:
            Dictionary with MACs and params for each text encoder, totals, and token length
        """
        if not tokenized_inputs_all:
            raise ValueError("tokenized_inputs_all cannot be empty. Must provide real tokenized inputs from pipeline.")
        
        results = {
            "text_encoder_1_macs": 0,
            "text_encoder_2_macs": 0,
            "text_encoder_3_macs": 0,
            "text_encoder_4_macs": 0,
            "text_encoder_1_params": 0,
            "text_encoder_2_params": 0,
            "text_encoder_3_params": 0,
            "text_encoder_4_params": 0,
            "total_macs": 0,
            "total_params": 0,
            "max_token_length": 0,
        }
        
        # Profile text_encoder (primary)
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            if 'text_encoder_1' not in tokenized_inputs_all:
                raise ValueError("tokenized_inputs_all must contain 'text_encoder_1' key for text_encoder")
            macs, params, max_len = self.profile(
                pipe.text_encoder, tokenized_inputs_all['text_encoder_1'], arch_config, verbose
            )
            results["text_encoder_1_macs"] = macs
            results["text_encoder_1_params"] = params
            results["total_macs"] += macs
            results["total_params"] += params
            results["max_token_length"] = max(results["max_token_length"], max_len)
        
        # Profile text_encoder_2
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            if 'text_encoder_2' not in tokenized_inputs_all:
                raise ValueError("tokenized_inputs_all must contain 'text_encoder_2' key for text_encoder_2")
            macs, params, max_len = self.profile(
                pipe.text_encoder_2, tokenized_inputs_all['text_encoder_2'], arch_config, verbose
            )
            results["text_encoder_2_macs"] = macs
            results["text_encoder_2_params"] = params
            results["total_macs"] += macs
            results["total_params"] += params
            results["max_token_length"] = max(results["max_token_length"], max_len)
        
        # Profile text_encoder_3
        if hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
            if 'text_encoder_3' not in tokenized_inputs_all:
                raise ValueError("tokenized_inputs_all must contain 'text_encoder_3' key for text_encoder_3")
            macs, params, max_len = self.profile(
                pipe.text_encoder_3, tokenized_inputs_all['text_encoder_3'], arch_config, verbose
            )
            results["text_encoder_3_macs"] = macs
            results["text_encoder_3_params"] = params
            results["total_macs"] += macs
            results["total_params"] += params
            results["max_token_length"] = max(results["max_token_length"], max_len)
        
        # Profile text_encoder_4
        if hasattr(pipe, 'text_encoder_4') and pipe.text_encoder_4 is not None:
            if 'text_encoder_4' not in tokenized_inputs_all:
                raise ValueError("tokenized_inputs_all must contain 'text_encoder_4' key for text_encoder_4")
            macs, params, max_len = self.profile(
                pipe.text_encoder_4, tokenized_inputs_all['text_encoder_4'], arch_config, verbose
            )
            results["text_encoder_4_macs"] = macs
            results["text_encoder_4_params"] = params
            results["total_macs"] += macs
            results["total_params"] += params
            results["max_token_length"] = max(results["max_token_length"], max_len)
        
        return results

