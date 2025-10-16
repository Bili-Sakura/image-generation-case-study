"""
Compute profiler for calculating FLOPs and MACs during inference.
Rewritten to use thop library with SDPA handling.
Uses model configuration from config.py for component detection.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from diffusers import DiffusionPipeline
import torch.nn.functional as F
from .config import MODELS


# --- SDPA FLOPs helper (approx) ---
# SDPA (QK^T softmax V) ~ 2 * B * heads * N^2 * head_dim + B * heads * N^2 (softmax) + B * heads * N * N (attn * V) * head_dim
# We'll register a lightweight counter for torch.nn.functional.scaled_dot_product_attention if present.

def sdpa_flops(q, k, v):
    """Calculate FLOPs for scaled dot product attention.
    
    Args:
        q, k, v: Query, key, value tensors [B, heads, N, head_dim]
    
    Returns:
        Total MACs for SDPA operation
    """
    # q,k,v: [B, heads, N, head_dim]
    B, H, N, D = q.shape
    # qk^T: B*H*N*N*D  (MACs: N*N*D; FLOPs ≈ 2x MACs; we stick to MACs)
    mac_qk = B * H * N * N * D
    # softmax: ~B*H*N*N  (cheap; include as MACs≈N*N)
    mac_softmax = B * H * N * N
    # attn @ v: B*H*N*N*D
    mac_av = B * H * N * N * D
    return mac_qk + mac_softmax + mac_av


# Monkeypatch wrapper to count SDPA MACs during a profiling pass
_sdpa_macs_counter = {"macs": 0}
_orig_sdpa = None


def _wrap_sdpa_for_macs():
    """Wrap SDPA function to count MACs during profiling."""
    global _orig_sdpa
    if _orig_sdpa is not None:
        return
    _orig_sdpa = F.scaled_dot_product_attention
    
    def wrapped(q, k, v, *args, **kwargs):
        _sdpa_macs_counter["macs"] += sdpa_flops(q, k, v)
        return _orig_sdpa(q, k, v, *args, **kwargs)
    
    F.scaled_dot_product_attention = wrapped


def _unwrap_sdpa():
    """Restore original SDPA function."""
    global _orig_sdpa
    if _orig_sdpa is not None:
        F.scaled_dot_product_attention = _orig_sdpa
        _orig_sdpa = None


# Wrapper modules for thop profiling
class UNetWrapper(nn.Module):
    """Wrapper for standard UNet (SD 1.x, 2.x, SDXL) to make it compatible with thop.profile()"""
    def __init__(self, unet, timestep, encoder_hidden_states):
        super().__init__()
        self.unet = unet
        self.timestep = timestep
        self.encoder_hidden_states = encoder_hidden_states
    
    def forward(self, x):
        return self.unet(x, self.timestep, encoder_hidden_states=self.encoder_hidden_states, return_dict=False)[0]


class FluxTransformerWrapper(nn.Module):
    """Wrapper for FLUX transformer to make it compatible with thop.profile()"""
    def __init__(self, transformer, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance=None):
        super().__init__()
        self.transformer = transformer
        self.encoder_hidden_states = encoder_hidden_states
        self.pooled_projections = pooled_projections
        self.timestep = timestep
        self.img_ids = img_ids
        self.txt_ids = txt_ids
        self.guidance = guidance
    
    def forward(self, hidden_states):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            pooled_projections=self.pooled_projections,
            timestep=self.timestep,
            img_ids=self.img_ids,
            txt_ids=self.txt_ids,
            guidance=self.guidance,
            return_dict=False
        )[0]


class SD3TransformerWrapper(nn.Module):
    """Wrapper for SD3 transformer to make it compatible with thop.profile()"""
    def __init__(self, transformer, encoder_hidden_states, pooled_projections, timestep):
        super().__init__()
        self.transformer = transformer
        self.encoder_hidden_states = encoder_hidden_states
        self.pooled_projections = pooled_projections
        self.timestep = timestep
    
    def forward(self, hidden_states):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            pooled_projections=self.pooled_projections,
            timestep=self.timestep,
            return_dict=False
        )[0]


class GenericTransformerWrapper(nn.Module):
    """
    Generic wrapper for transformers where hidden_states is the first positional argument.
    Works for: SANA, PixArt, QwenImage, CogView4, and similar architectures.
    """
    def __init__(self, transformer, encoder_hidden_states, timestep):
        super().__init__()
        self.transformer = transformer
        self.encoder_hidden_states = encoder_hidden_states
        self.timestep = timestep
    
    def forward(self, hidden_states):
        # These transformers expect: forward(hidden_states, encoder_hidden_states, timestep, ...)
        return self.transformer(
            hidden_states,  # First positional argument
            self.encoder_hidden_states,  # Second positional argument
            self.timestep,   # Third positional argument
            return_dict=False
        )[0]


class LuminaNextTransformerWrapper(nn.Module):
    """Wrapper for LuminaNext transformer (needs encoder_mask and image_rotary_emb)"""
    def __init__(self, transformer, encoder_hidden_states, timestep, encoder_mask, image_rotary_emb):
        super().__init__()
        self.transformer = transformer
        self.encoder_hidden_states = encoder_hidden_states
        self.timestep = timestep
        self.encoder_mask = encoder_mask
        self.image_rotary_emb = image_rotary_emb
    
    def forward(self, hidden_states):
        return self.transformer(
            hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_mask,
            self.image_rotary_emb,
            cross_attention_kwargs={},  # Needed to avoid "argument after ** must be a mapping" error
            return_dict=False
        )[0]


class Lumina2TransformerWrapper(nn.Module):
    """Wrapper for Lumina2 transformer (needs encoder_attention_mask)"""
    def __init__(self, transformer, encoder_hidden_states, timestep, encoder_attention_mask):
        super().__init__()
        self.transformer = transformer
        self.encoder_hidden_states = encoder_hidden_states
        self.timestep = timestep
        self.encoder_attention_mask = encoder_attention_mask
    
    def forward(self, hidden_states):
        return self.transformer(
            hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_attention_mask,
            return_dict=False
        )[0]


class KandinskyUNetWrapper(nn.Module):
    """Wrapper for Kandinsky-3 UNet (needs encoder_attention_mask)"""
    def __init__(self, unet, timestep, encoder_hidden_states, encoder_attention_mask):
        super().__init__()
        self.unet = unet
        self.timestep = timestep
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_attention_mask = encoder_attention_mask
    
    def forward(self, sample):
        return self.unet(
            sample,
            self.timestep,
            encoder_hidden_states=self.encoder_hidden_states,
            encoder_attention_mask=self.encoder_attention_mask,
            return_dict=False
        )[0]


class Kandinsky2UNetWrapper(nn.Module):
    """Wrapper for Kandinsky-2.1 UNet (simpler than Kandinsky-3)"""
    def __init__(self, unet, timestep, encoder_hidden_states):
        super().__init__()
        self.unet = unet
        self.timestep = timestep
        self.encoder_hidden_states = encoder_hidden_states
    
    def forward(self, sample):
        return self.unet(
            sample,
            self.timestep,
            encoder_hidden_states=self.encoder_hidden_states,
            return_dict=False
        )[0]


class VAEWrapper(nn.Module):
    """Wrapper for VAE decoder to make it compatible with thop.profile()"""
    def __init__(self, vae, scaling_factor):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor
    
    def forward(self, z):
        return self.vae.decode(z / self.scaling_factor, return_dict=False)[0]


class TextEncoderWrapper(nn.Module):
    """Wrapper for text encoder to make it compatible with thop.profile()"""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
    
    def forward(self, input_ids):
        return self.text_encoder(input_ids, return_dict=False)[0]


class ComputeProfiler:
    """Profiler for measuring FLOPs, MACs, and inference time using thop."""
    
    def __init__(self, enabled: bool = True):
        """Initialize the compute profiler.
        
        Args:
            enabled: Whether to enable profiling (can be disabled for performance).
        """
        self.enabled = enabled
        self._thop_available = False
        
        if self.enabled:
            try:
                from thop import profile
                self._thop_available = True
                self._profile = profile
            except ImportError:
                print("⚠️  thop not installed. Install with: pip install thop")
                print("   Alternative: pip install thop==0.1.1-2209072238")
                print("   FLOPs/MACs profiling will be disabled.")
                self.enabled = False
    
    def get_model_config(self, model_id: str) -> Optional[Dict]:
        """Get model configuration from config.py.
        
        Args:
            model_id: Model identifier (e.g., "stabilityai/stable-diffusion-2-1-base")
            
        Returns:
            Model configuration dictionary or None if not found
        """
        return MODELS.get(model_id)
    
    def detect_model_components(self, pipe, model_id: Optional[str] = None) -> Dict[str, any]:
        """Detect what components a model has using config.py and pipeline inspection.
        
        Args:
            pipe: The diffusion pipeline
            model_id: Optional model identifier to look up config
            
        Returns:
            Dictionary with component information
        """
        components = {
            "has_unet": False,
            "has_transformer": False,
            "has_vae": False,
            "has_text_encoder": False,
            "has_text_encoder_2": False,
            "has_text_encoder_3": False,
            "has_text_encoder_4": False,
            "num_text_encoders": 0,
            "has_tokenizer": False,
            "has_tokenizer_2": False,
            "has_tokenizer_3": False,
            "has_tokenizer_4": False,
            "main_model_type": None,
            "main_model": None,
            "model_config": None,
            "pipeline_class": None,
        }
        
        # Get config if model_id is provided
        if model_id:
            config = self.get_model_config(model_id)
            if config:
                components["model_config"] = config
                components["pipeline_class"] = config.get("pipeline_class")
        
        # Check for UNet
        if hasattr(pipe, 'unet') and pipe.unet is not None:
            components["has_unet"] = True
            components["main_model_type"] = "unet"
            components["main_model"] = pipe.unet
        
        # Check for Transformer
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            components["has_transformer"] = True
            components["main_model_type"] = "transformer"
            components["main_model"] = pipe.transformer
        
        # Check for VAE
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            components["has_vae"] = True
        
        # Check for text encoders (up to 4)
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            components["has_text_encoder"] = True
            components["num_text_encoders"] += 1
        
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            components["has_text_encoder_2"] = True
            components["num_text_encoders"] += 1
        
        if hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
            components["has_text_encoder_3"] = True
            components["num_text_encoders"] += 1
        
        if hasattr(pipe, 'text_encoder_4') and pipe.text_encoder_4 is not None:
            components["has_text_encoder_4"] = True
            components["num_text_encoders"] += 1
        
        # Check for tokenizers (up to 4)
        if hasattr(pipe, 'tokenizer') and pipe.tokenizer is not None:
            components["has_tokenizer"] = True
        
        if hasattr(pipe, 'tokenizer_2') and pipe.tokenizer_2 is not None:
            components["has_tokenizer_2"] = True
        
        if hasattr(pipe, 'tokenizer_3') and pipe.tokenizer_3 is not None:
            components["has_tokenizer_3"] = True
        
        if hasattr(pipe, 'tokenizer_4') and pipe.tokenizer_4 is not None:
            components["has_tokenizer_4"] = True
        
        return components
    
    def _encode_text(self, pipe, prompt: str, negative_prompt: str = ""):
        """Encode text prompts using the pipeline's text encoder.
        
        Args:
            pipe: The diffusion pipeline
            prompt: The main prompt
            negative_prompt: The negative prompt
            
        Returns:
            Tuple of (conditional_embeddings, unconditional_embeddings)
        """
        device = pipe.device if hasattr(pipe, 'device') else next(pipe.text_encoder.parameters()).device
        tokenizer = pipe.tokenizer
        
        enc = tokenizer(
            [prompt], 
            padding="max_length", 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(device)
        
        neg = tokenizer(
            [negative_prompt], 
            padding="max_length", 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            te = pipe.text_encoder(enc.input_ids)[0]
            te_neg = pipe.text_encoder(neg.input_ids)[0]
        
        return te, te_neg
    
    def _get_model_type(self, model, model_config: Optional[Dict] = None):
        """Detect the type of model using config and class name.
        
        Args:
            model: The model instance
            model_config: Optional model configuration from config.py
            
        Returns:
            Model type string for wrapper selection
        """
        model_class_name = model.__class__.__name__
        
        # Try to use config first if available
        if model_config:
            pipeline_class = model_config.get("pipeline_class", "")
            
            # Map pipeline classes to model types
            if "Flux" in pipeline_class:
                return 'flux'
            elif "StableDiffusion3" in pipeline_class:
                return 'sd3'
            elif "LuminaText2Img" in pipeline_class or "LuminaNext" in pipeline_class:
                return 'lumina_next'
            elif "Lumina2" in pipeline_class:
                return 'lumina2'
            elif "Kandinsky3" in pipeline_class:
                return 'kandinsky3'
            elif "Kandinsky" in pipeline_class and "2" in pipeline_class:
                return 'kandinsky2'
            elif any(name in pipeline_class for name in ['Sana', 'PixArt', 'Qwen', 'CogView', 'HiDream']):
                return 'generic_transformer'
            elif "UNet" in pipeline_class or "StableDiffusion" in pipeline_class:
                return 'unet'
        
        # Fallback to class name detection
        if 'Flux' in model_class_name:
            return 'flux'
        elif 'SD3' in model_class_name:
            return 'sd3'
        elif 'LuminaNext' in model_class_name:
            return 'lumina_next'
        elif 'Lumina2' in model_class_name:
            return 'lumina2'
        elif 'Kandinsky3' in model_class_name:
            return 'kandinsky3'
        elif 'Kandinsky2' in model_class_name or 'Kandinsky' in model_class_name:
            return 'kandinsky2'
        elif any(name in model_class_name for name in ['Sana', 'PixArt', 'Qwen', 'CogView', 'HiDream']):
            return 'generic_transformer'
        elif 'UNet' in model_class_name:
            return 'unet'
        else:
            return 'unknown'
    
    def _prepare_transformer_inputs(self, pipe, model, model_type, height, width, prompt, device, dtype):
        """Prepare inputs for transformer models (FLUX, SD3)."""
        h8, w8 = height // 8, width // 8
        
        # Get latent channels from VAE config
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'config'):
            latent_channels = getattr(pipe.vae.config, 'latent_channels', 16)
        else:
            latent_channels = 16
        
        # For FLUX, sample stays in NCHW format; for SD3, it's converted to sequence format internally
        sample = torch.randn(1, latent_channels, h8, w8, device=device, dtype=dtype)
        t = torch.tensor([500], device=device, dtype=torch.long)
        
        # Encode text to get proper embeddings
        # Try to get the correct embedding dimension from config
        if hasattr(model, 'config'):
            # Try different config attributes for embedding dimension
            joint_attention_dim = getattr(model.config, 'joint_attention_dim', None)
            if joint_attention_dim is None:
                joint_attention_dim = getattr(model.config, 'caption_channels', None)  # SANA
            if joint_attention_dim is None:
                joint_attention_dim = getattr(model.config, 'cap_feat_dim', None)  # Lumina2
            if joint_attention_dim is None:
                joint_attention_dim = getattr(model.config, 'cross_attention_dim', None)  # PixArt
            if joint_attention_dim is None:
                joint_attention_dim = 4096  # Default fallback
        else:
            joint_attention_dim = 4096
        
        # Text sequence length
        text_seq_len = 77
        encoder_hidden_states = torch.randn(1, text_seq_len, joint_attention_dim, device=device, dtype=dtype)
        
        # Pooled projections
        if hasattr(model, 'config'):
            pooled_projection_dim = getattr(model.config, 'pooled_projection_dim', 768)
        else:
            pooled_projection_dim = 768
        pooled_projections = torch.randn(1, pooled_projection_dim, device=device, dtype=dtype)
        
        if model_type == 'flux':
            # FLUX-specific inputs
            # img_ids and txt_ids for positional encoding (2D tensors without batch dimension)
            img_ids = torch.zeros(h8 * w8, 3, device=device, dtype=torch.float32)
            txt_ids = torch.zeros(text_seq_len, 3, device=device, dtype=torch.float32)
            
            # Check if model uses guidance embeddings
            guidance = None
            if hasattr(model, 'config') and hasattr(model.config, 'guidance_embeds'):
                if model.config.guidance_embeds:
                    guidance = torch.tensor([3.5], device=device, dtype=torch.float32)
            
            return {
                'sample': sample,  # FLUX expects (B, C, H, W)
                'encoder_hidden_states': encoder_hidden_states,
                'pooled_projections': pooled_projections,
                'timestep': t,
                'img_ids': img_ids,  # 2D: (img_seq_len, 3)
                'txt_ids': txt_ids,  # 2D: (text_seq_len, 3)
                'guidance': guidance,
            }
        elif model_type == 'sd3':
            # SD3 inputs (sample should also be in NCHW format)
            return {
                'sample': sample,  # SD3 also expects (B, C, H, W)
                'encoder_hidden_states': encoder_hidden_states,
                'pooled_projections': pooled_projections,
                'timestep': t,
            }
        elif model_type == 'generic_transformer':
            # Generic transformer (SANA, PixArt, Qwen, CogView, etc.)
            # These models typically don't need pooled_projections
            return {
                'sample': sample,
                'encoder_hidden_states': encoder_hidden_states,
                'timestep': t,
            }
        elif model_type == 'lumina_next':
            # LuminaNext needs encoder_mask and image_rotary_emb
            encoder_mask = torch.ones(1, text_seq_len, device=device, dtype=torch.bool)
            # Create simple image rotary embeddings
            image_rotary_emb = torch.zeros(h8 * w8, 256, device=device, dtype=dtype)  # Typical dim
            return {
                'sample': sample,
                'encoder_hidden_states': encoder_hidden_states,
                'timestep': t,
                'encoder_mask': encoder_mask,
                'image_rotary_emb': image_rotary_emb,
            }
        elif model_type == 'lumina2':
            # Lumina2 needs encoder_attention_mask
            encoder_attention_mask = torch.ones(1, text_seq_len, device=device, dtype=torch.long)
            return {
                'sample': sample,
                'encoder_hidden_states': encoder_hidden_states,
                'timestep': t,
                'encoder_attention_mask': encoder_attention_mask,
            }
        else:
            return None

    def measure_unet_macs(
        self, 
        pipe, 
        height: int = 512, 
        width: int = 512, 
        prompt: str = "a photo of a cat", 
        guidance_scale: float = 7.5,
        model_id: Optional[str] = None
    ) -> int:
        """Measure MACs for UNet/Transformer forward pass.
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
            prompt: Text prompt for conditioning
            guidance_scale: Guidance scale (affects whether we do 1 or 2 passes)
            model_id: Optional model identifier to look up config
            
        Returns:
            MACs per inference step
        """
        if not self._thop_available:
            return 0
        
        # Get the UNet/Transformer model
        model = None
        if hasattr(pipe, 'unet') and pipe.unet is not None:
            model = pipe.unet
        elif hasattr(pipe, 'transformer') and pipe.transformer is not None:
            model = pipe.transformer
        
        if model is None:
            return 0
        
        model.eval()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Get model config
        model_config = self.get_model_config(model_id) if model_id else None
        
        # Detect model type
        model_type = self._get_model_type(model, model_config)
        
        with torch.no_grad():
            if model_type in ['flux', 'sd3', 'generic_transformer', 'lumina_next', 'lumina2']:
                # Handle transformer models
                inputs = self._prepare_transformer_inputs(pipe, model, model_type, height, width, prompt, device, dtype)
                if inputs is None:
                    return 0
                
                # Wrap SDPA for attention MAC counting
                _wrap_sdpa_for_macs()
                
                try:
                    # Create appropriate wrapper
                    if model_type == 'flux':
                        wrapper = FluxTransformerWrapper(
                            model,
                            inputs['encoder_hidden_states'],
                            inputs['pooled_projections'],
                            inputs['timestep'],
                            inputs['img_ids'],
                            inputs['txt_ids'],
                            inputs['guidance']
                        ).to(device)
                    elif model_type == 'sd3':
                        wrapper = SD3TransformerWrapper(
                            model,
                            inputs['encoder_hidden_states'],
                            inputs['pooled_projections'],
                            inputs['timestep']
                        ).to(device)
                    elif model_type == 'lumina_next':
                        wrapper = LuminaNextTransformerWrapper(
                            model,
                            inputs['encoder_hidden_states'],
                            inputs['timestep'],
                            inputs['encoder_mask'],
                            inputs['image_rotary_emb']
                        ).to(device)
                    elif model_type == 'lumina2':
                        wrapper = Lumina2TransformerWrapper(
                            model,
                            inputs['encoder_hidden_states'],
                            inputs['timestep'],
                            inputs['encoder_attention_mask']
                        ).to(device)
                    else:  # generic_transformer (SANA, PixArt, Qwen, CogView, etc.)
                        wrapper = GenericTransformerWrapper(
                            model,
                            inputs['encoder_hidden_states'],
                            inputs['timestep']
                        ).to(device)
                    
                    wrapper.eval()
                    macs, _ = self._profile(wrapper, inputs=(inputs['sample'],), verbose=False)
                except Exception as e:
                    print(f"⚠️  Error profiling {model_type.upper()} transformer: {e}")
                    import traceback
                    traceback.print_exc()
                    macs = 0
                
                sdpa_macs = _sdpa_macs_counter["macs"]
                _sdpa_macs_counter["macs"] = 0
                _unwrap_sdpa()
                
                # Add SDPA MACs
                macs += sdpa_macs
                
                # For transformers, guidance_scale determines if we do CFG (2x passes)
                # FLUX.1-dev typically doesn't use CFG, but check guidance_scale
                per_step = macs * (2 if guidance_scale and guidance_scale > 1.0 else 1)
                return int(per_step)
            
            elif model_type == 'kandinsky3':
                # Kandinsky-3 UNet needs encoder_attention_mask
                h8, w8 = height // 8, width // 8
                
                # Get latent channels
                if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'config'):
                    latent_channels = getattr(pipe.vae.config, 'latent_channels', 4)
                else:
                    latent_channels = 4
                
                sample = torch.randn(1, latent_channels, h8, w8, device=device, dtype=dtype)
                t = torch.tensor([500], device=device, dtype=torch.long)
                
                # Try to encode text
                try:
                    cond, uncond = self._encode_text(pipe, prompt, "")
                except:
                    if hasattr(model, 'config'):
                        encoder_dim = getattr(model.config, 'encoder_hid_dim', 1024)
                    else:
                        encoder_dim = 1024
                    cond = torch.randn(1, 77, encoder_dim, device=device, dtype=dtype)
                    uncond = torch.randn(1, 77, encoder_dim, device=device, dtype=dtype)
                
                # Create encoder_attention_mask
                encoder_attention_mask = torch.ones(1, 77, device=device, dtype=torch.long)
                
                # Wrap SDPA
                _wrap_sdpa_for_macs()
                
                # Profile with encoder_attention_mask
                kandinsky_wrapper_cond = KandinskyUNetWrapper(model, t, cond, encoder_attention_mask).to(device)
                kandinsky_wrapper_cond.eval()
                
                try:
                    macs_cond, _ = self._profile(kandinsky_wrapper_cond, inputs=(sample.clone(),), verbose=False)
                except Exception as e:
                    print(f"⚠️  Error profiling Kandinsky conditional pass: {e}")
                    import traceback
                    traceback.print_exc()
                    macs_cond = 0
                
                kandinsky_wrapper_uncond = KandinskyUNetWrapper(model, t, uncond, encoder_attention_mask).to(device)
                kandinsky_wrapper_uncond.eval()
                
                try:
                    macs_uncond, _ = self._profile(kandinsky_wrapper_uncond, inputs=(sample.clone(),), verbose=False)
                except Exception as e:
                    print(f"⚠️  Error profiling Kandinsky unconditional pass: {e}")
                    import traceback
                    traceback.print_exc()
                    macs_uncond = 0
                
                sdpa_macs = _sdpa_macs_counter["macs"]
                _sdpa_macs_counter["macs"] = 0
                _unwrap_sdpa()
                
                macs_single = (macs_cond + macs_uncond) / 2
                macs_single += sdpa_macs
                per_step = macs_single * (2 if guidance_scale and guidance_scale > 1.0 else 1)
                return int(per_step)
            
            elif model_type == 'kandinsky2':
                # Kandinsky-2.1 UNet (simpler, no encoder_attention_mask)
                h8, w8 = height // 8, width // 8
                
                # Get latent channels
                if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'config'):
                    latent_channels = getattr(pipe.vae.config, 'latent_channels', 4)
                else:
                    latent_channels = 4
                
                sample = torch.randn(1, latent_channels, h8, w8, device=device, dtype=dtype)
                t = torch.tensor([500], device=device, dtype=torch.long)
                
                # Try to encode text
                try:
                    cond, uncond = self._encode_text(pipe, prompt, "")
                except:
                    if hasattr(model, 'config'):
                        encoder_dim = getattr(model.config, 'encoder_hid_dim', 768)
                    else:
                        encoder_dim = 768
                    cond = torch.randn(1, 77, encoder_dim, device=device, dtype=dtype)
                    uncond = torch.randn(1, 77, encoder_dim, device=device, dtype=dtype)
                
                # Wrap SDPA
                _wrap_sdpa_for_macs()
                
                # Profile Kandinsky 2.1 (no encoder_attention_mask needed)
                kandinsky2_wrapper_cond = Kandinsky2UNetWrapper(model, t, cond).to(device)
                kandinsky2_wrapper_cond.eval()
                
                try:
                    macs_cond, _ = self._profile(kandinsky2_wrapper_cond, inputs=(sample.clone(),), verbose=False)
                except Exception as e:
                    print(f"⚠️  Error profiling Kandinsky-2 conditional pass: {e}")
                    import traceback
                    traceback.print_exc()
                    macs_cond = 0
                
                kandinsky2_wrapper_uncond = Kandinsky2UNetWrapper(model, t, uncond).to(device)
                kandinsky2_wrapper_uncond.eval()
                
                try:
                    macs_uncond, _ = self._profile(kandinsky2_wrapper_uncond, inputs=(sample.clone(),), verbose=False)
                except Exception as e:
                    print(f"⚠️  Error profiling Kandinsky-2 unconditional pass: {e}")
                    import traceback
                    traceback.print_exc()
                    macs_uncond = 0
                
                sdpa_macs = _sdpa_macs_counter["macs"]
                _sdpa_macs_counter["macs"] = 0
                _unwrap_sdpa()
                
                macs_single = (macs_cond + macs_uncond) / 2
                macs_single += sdpa_macs
                per_step = macs_single * (2 if guidance_scale and guidance_scale > 1.0 else 1)
                return int(per_step)
            
            else:
                # Handle standard UNet models (SD 1.x, 2.x, SDXL)
                h8, w8 = height // 8, width // 8
                
                # Get latent channels from VAE config
                if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'config'):
                    latent_channels = getattr(pipe.vae.config, 'latent_channels', 4)
                else:
                    latent_channels = 4
                
                sample = torch.randn(1, latent_channels, h8, w8, device=device, dtype=dtype)
                t = torch.tensor([500], device=device, dtype=torch.long)
                
                # Try to encode text if text encoder is available
                try:
                    cond, uncond = self._encode_text(pipe, prompt, "")
                except:
                    # Fallback: use dummy embeddings
                    if hasattr(model, 'config'):
                        encoder_dim = getattr(model.config, 'cross_attention_dim', 1024)
                    else:
                        encoder_dim = 1024
                    cond = torch.randn(1, 77, encoder_dim, device=device, dtype=dtype)
                    uncond = torch.randn(1, 77, encoder_dim, device=device, dtype=dtype)
                
                # Wrap SDPA for attention MAC counting
                _wrap_sdpa_for_macs()
                
                # Profile one UNet pass (conditional) using wrapper module
                unet_wrapper_cond = UNetWrapper(model, t, cond).to(device)
                unet_wrapper_cond.eval()
                
                try:
                    macs_cond, _ = self._profile(unet_wrapper_cond, inputs=(sample.clone(),), verbose=False)
                except Exception as e:
                    print(f"⚠️  Error profiling UNet conditional pass: {e}")
                    import traceback
                    traceback.print_exc()
                    macs_cond = 0
                
                # Profile one UNet pass (unconditional) using wrapper module
                unet_wrapper_uncond = UNetWrapper(model, t, uncond).to(device)
                unet_wrapper_uncond.eval()
                
                try:
                    macs_uncond, _ = self._profile(unet_wrapper_uncond, inputs=(sample.clone(),), verbose=False)
                except Exception as e:
                    print(f"⚠️  Error profiling UNet unconditional pass: {e}")
                    import traceback
                    traceback.print_exc()
                    macs_uncond = 0
                
                sdpa_macs = _sdpa_macs_counter["macs"]
                _sdpa_macs_counter["macs"] = 0
                _unwrap_sdpa()
                
                # THOP counts linear/conv MACs well; SDPA is undercounted unless we add it.
                macs_unet_single = (macs_cond + macs_uncond) / 2  # average single pass
                macs_unet_single += sdpa_macs  # add SDPA contribution once (approx per pass)
                
                # If CFG>1, per step we do two UNet passes
                per_step = macs_unet_single * (2 if guidance_scale and guidance_scale > 1.0 else 1)
                return int(per_step)
    
    def measure_vae_decode_macs(self, pipe, height: int = 512, width: int = 512) -> int:
        """Measure MACs for VAE decoder.
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
            
        Returns:
            MACs for VAE decode operation
        """
        if not self._thop_available or not hasattr(pipe, 'vae'):
            return 0
        
        pipe.vae.eval()
        device = next(pipe.vae.parameters()).device
        dtype = next(pipe.vae.parameters()).dtype
        
        with torch.no_grad():
            h8, w8 = height // 8, width // 8
            
            # Get latent channels from VAE config (4 for SD, 16 for FLUX/SD3)
            latent_channels = getattr(pipe.vae.config, 'latent_channels', 4)
            latents = torch.randn(1, latent_channels, h8, w8, device=device, dtype=dtype)
            
            scaling_factor = getattr(pipe.vae.config, 'scaling_factor', 0.18215)
            vae_wrapper = VAEWrapper(pipe.vae, scaling_factor).to(device)
            vae_wrapper.eval()
            
            try:
                macs, _ = self._profile(vae_wrapper, inputs=(latents,), verbose=False)
                return int(macs)
            except Exception as e:
                print(f"⚠️  Error profiling VAE decoder: {e}")
                import traceback
                traceback.print_exc()
                return 0
    
    def measure_text_encoder_macs(
        self, 
        pipe, 
        prompt: str = "a photo of a cat", 
        negative_prompt: str = "",
        model_id: Optional[str] = None
    ) -> Dict[str, int]:
        """Measure MACs for text encoder(s).
        
        Args:
            pipe: The diffusion pipeline
            prompt: Text prompt
            negative_prompt: Negative text prompt
            model_id: Optional model identifier to look up config
            
        Returns:
            Dictionary with MACs for each text encoder
        """
        if not self._thop_available:
            return {"total": 0, "text_encoder_1": 0, "text_encoder_2": 0, "text_encoder_3": 0, "text_encoder_4": 0}
        
        components = self.detect_model_components(pipe, model_id)
        total_macs = 0
        result = {"text_encoder_1": 0, "text_encoder_2": 0, "text_encoder_3": 0, "text_encoder_4": 0}
        
        # Profile text_encoder (primary)
        if components["has_text_encoder"]:
            try:
                device = next(pipe.text_encoder.parameters()).device
                tokenizer = pipe.tokenizer
                
                with torch.no_grad():
                    enc = tokenizer(
                        [prompt], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=77, 
                        return_tensors="pt"
                    ).to(device)
                    
                    neg = tokenizer(
                        [negative_prompt], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=77, 
                        return_tensors="pt"
                    ).to(device)
                    
                    te_wrapper = TextEncoderWrapper(pipe.text_encoder).to(device)
                    te_wrapper.eval()
                    
                    try:
                        macs1, _ = self._profile(te_wrapper, inputs=(enc.input_ids,), verbose=False)
                        macs2, _ = self._profile(te_wrapper, inputs=(neg.input_ids,), verbose=False)
                        result["text_encoder_1"] = int(macs1 + macs2)
                        total_macs += result["text_encoder_1"]
                    except Exception as e:
                        print(f"⚠️  Error profiling text encoder 1: {e}")
            except Exception as e:
                print(f"⚠️  Could not profile text encoder 1: {e}")
        
        # Profile text_encoder_2 (e.g., SDXL, SD3)
        if components["has_text_encoder_2"]:
            try:
                device = next(pipe.text_encoder_2.parameters()).device
                tokenizer = pipe.tokenizer_2 if components["has_tokenizer_2"] else pipe.tokenizer
                
                with torch.no_grad():
                    enc = tokenizer(
                        [prompt], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=77, 
                        return_tensors="pt"
                    ).to(device)
                    
                    neg = tokenizer(
                        [negative_prompt], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=77, 
                        return_tensors="pt"
                    ).to(device)
                    
                    te_wrapper = TextEncoderWrapper(pipe.text_encoder_2).to(device)
                    te_wrapper.eval()
                    
                    try:
                        macs1, _ = self._profile(te_wrapper, inputs=(enc.input_ids,), verbose=False)
                        macs2, _ = self._profile(te_wrapper, inputs=(neg.input_ids,), verbose=False)
                        result["text_encoder_2"] = int(macs1 + macs2)
                        total_macs += result["text_encoder_2"]
                    except Exception as e:
                        print(f"⚠️  Error profiling text encoder 2: {e}")
            except Exception as e:
                print(f"⚠️  Could not profile text encoder 2: {e}")
        
        # Profile text_encoder_3 (e.g., SD3, FLUX)
        if components["has_text_encoder_3"]:
            try:
                device = next(pipe.text_encoder_3.parameters()).device
                tokenizer = pipe.tokenizer_3 if components["has_tokenizer_3"] else pipe.tokenizer
                
                with torch.no_grad():
                    enc = tokenizer(
                        [prompt], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=77, 
                        return_tensors="pt"
                    ).to(device)
                    
                    neg = tokenizer(
                        [negative_prompt], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=77, 
                        return_tensors="pt"
                    ).to(device)
                    
                    te_wrapper = TextEncoderWrapper(pipe.text_encoder_3).to(device)
                    te_wrapper.eval()
                    
                    try:
                        macs1, _ = self._profile(te_wrapper, inputs=(enc.input_ids,), verbose=False)
                        macs2, _ = self._profile(te_wrapper, inputs=(neg.input_ids,), verbose=False)
                        result["text_encoder_3"] = int(macs1 + macs2)
                        total_macs += result["text_encoder_3"]
                    except Exception as e:
                        print(f"⚠️  Error profiling text encoder 3: {e}")
            except Exception as e:
                print(f"⚠️  Could not profile text encoder 3: {e}")
        
        # Profile text_encoder_4 (e.g., HiDream-I1-Full)
        if components["has_text_encoder_4"]:
            try:
                device = next(pipe.text_encoder_4.parameters()).device
                tokenizer = pipe.tokenizer_4 if components["has_tokenizer_4"] else pipe.tokenizer
                
                with torch.no_grad():
                    enc = tokenizer(
                        [prompt], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=77, 
                        return_tensors="pt"
                    ).to(device)
                    
                    neg = tokenizer(
                        [negative_prompt], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=77, 
                        return_tensors="pt"
                    ).to(device)
                    
                    te_wrapper = TextEncoderWrapper(pipe.text_encoder_4).to(device)
                    te_wrapper.eval()
                    
                    try:
                        macs1, _ = self._profile(te_wrapper, inputs=(enc.input_ids,), verbose=False)
                        macs2, _ = self._profile(te_wrapper, inputs=(neg.input_ids,), verbose=False)
                        result["text_encoder_4"] = int(macs1 + macs2)
                        total_macs += result["text_encoder_4"]
                    except Exception as e:
                        print(f"⚠️  Error profiling text encoder 4: {e}")
            except Exception as e:
                print(f"⚠️  Could not profile text encoder 4: {e}")
        
        result["total"] = total_macs
        return result
    
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
        """Summarize MACs for complete image generation.
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
            steps: Number of inference steps
            prompt: Text prompt
            guidance_scale: Guidance scale
            model_id: Optional model identifier to look up config
            
        Returns:
            Dictionary with MAC counts and formatted strings
        """
        if not self.enabled or not self._thop_available:
            return {
                "enabled": False,
                "components": {},
                "UNet per-step (MACs)": 0,
                "UNet per-step (GMACs)": 0.0,
                "Text encoder once (GMACs)": 0.0,
                "VAE decode once (GMACs)": 0.0,
                f"Total {steps} steps (GMACs)": 0.0,
            }
        
        # First detect what components this model has
        components = self.detect_model_components(pipe, model_id)
        
        print(f"\n📋 Detected Model Components:")
        print(f"  Main Model: {components['main_model_type'] or 'None'}")
        if components.get('pipeline_class'):
            print(f"  Pipeline: {components['pipeline_class']}")
        if components['has_unet']:
            print(f"  ✓ UNet")
        if components['has_transformer']:
            print(f"  ✓ Transformer")
        if components['has_vae']:
            print(f"  ✓ VAE")
        if components['num_text_encoders'] > 0:
            print(f"  ✓ Text Encoders: {components['num_text_encoders']}")
            if components['has_text_encoder']:
                print(f"    - Text Encoder 1")
            if components['has_text_encoder_2']:
                print(f"    - Text Encoder 2")
            if components['has_text_encoder_3']:
                print(f"    - Text Encoder 3")
            if components['has_text_encoder_4']:
                print(f"    - Text Encoder 4")
        
        # Measure each component that exists
        m_unet_step = 0
        m_vae = 0
        m_text_total = 0
        text_encoder_details = {}
        
        # Main model (UNet/Transformer)
        if components['main_model'] is not None:
            print(f"\nProfiling {components['main_model_type']}...")
            m_unet_step = self.measure_unet_macs(pipe, height, width, prompt, guidance_scale, model_id)
        
        # VAE
        if components['has_vae']:
            print(f"Profiling VAE...")
            m_vae = self.measure_vae_decode_macs(pipe, height, width)
        
        # Text encoders
        if components['num_text_encoders'] > 0:
            print(f"Profiling text encoder(s)...")
            text_results = self.measure_text_encoder_macs(pipe, prompt, "", model_id)
            m_text_total = text_results["total"]
            text_encoder_details = {
                "text_encoder_1_macs": text_results["text_encoder_1"],
                "text_encoder_2_macs": text_results["text_encoder_2"],
                "text_encoder_3_macs": text_results["text_encoder_3"],
                "text_encoder_4_macs": text_results["text_encoder_4"],
            }
        
        total = m_unet_step * steps + m_vae + m_text_total
        
        to_gmac = lambda x: x / 1e9
        
        result = {
            "enabled": True,
            "components": components,
            "UNet per-step (MACs)": m_unet_step,
            "UNet per-step (GMACs)": round(to_gmac(m_unet_step), 3),
            "Text encoder once (GMACs)": round(to_gmac(m_text_total), 3),
            "VAE decode once (GMACs)": round(to_gmac(m_vae), 3),
            f"Total {steps} steps (GMACs)": round(to_gmac(total), 3),
        }
        
        # Add detailed text encoder breakdown
        if text_encoder_details:
            result.update({
                "Text encoder 1 (GMACs)": round(to_gmac(text_encoder_details["text_encoder_1_macs"]), 3),
                "Text encoder 2 (GMACs)": round(to_gmac(text_encoder_details["text_encoder_2_macs"]), 3),
                "Text encoder 3 (GMACs)": round(to_gmac(text_encoder_details["text_encoder_3_macs"]), 3),
                "Text encoder 4 (GMACs)": round(to_gmac(text_encoder_details["text_encoder_4_macs"]), 3),
            })
        
        return result
    
    def profile_pipeline(
        self,
        pipe: DiffusionPipeline,
        input_shape: Tuple[int, int, int, int],
        num_inference_steps: int = 50,
        model_id: str = "unknown",
        guidance_scale: float = 7.5,
    ) -> Dict[str, any]:
        """Profile a diffusion pipeline to calculate FLOPs and MACs.
        
        Args:
            pipe: The diffusion pipeline to profile
            input_shape: Input tensor shape (batch_size, channels, height, width)
            num_inference_steps: Number of inference steps
            model_id: Model identifier for reporting
            guidance_scale: Guidance scale for generation
            
        Returns:
            Dictionary containing FLOPs, MACs, parameters, and other metrics
        """
        if not self.enabled or not self._thop_available:
            return {
                "enabled": False,
                "total_flops": 0,
                "total_macs": 0,
                "total_params": 0,
                "flops_per_step": 0,
                "macs_per_step": 0,
            }
        
        try:
            # Extract dimensions
            batch_size, channels, height, width = input_shape
            
            # Get model component
            model_to_profile = None
            model_name = "model"
            
            if hasattr(pipe, 'transformer') and pipe.transformer is not None:
                model_to_profile = pipe.transformer
                model_name = "transformer"
            elif hasattr(pipe, 'unet') and pipe.unet is not None:
                model_to_profile = pipe.unet
                model_name = "unet"
            
            if model_to_profile is None:
                print(f"⚠️  Could not find transformer/unet in pipeline for {model_id}")
                return self._empty_profile()
            
            print(f"📊 Profiling {model_name} for {model_id}...")
            
            # Measure MACs for each component
            m_unet_step = self.measure_unet_macs(pipe, height, width, "a photo", guidance_scale, model_id)
            m_vae = self.measure_vae_decode_macs(pipe, height, width)
            m_text_results = self.measure_text_encoder_macs(pipe, "a photo", "", model_id)
            m_text = m_text_results["total"]
            
            # Calculate totals
            total_macs = m_unet_step * num_inference_steps + m_vae + m_text
            # FLOPs ≈ 2 × MACs (since each MAC involves multiply + add)
            total_flops = total_macs * 2
            flops_per_step = m_unet_step * 2
            macs_per_step = m_unet_step
            
            # Count parameters
            try:
                params = sum(p.numel() for p in model_to_profile.parameters())
            except:
                params = 0
            
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
                "latent_shape": (batch_size, 4 if model_name == "unet" else 16, height // 8, width // 8),
                # Human readable formats
                "total_flops_str": self._format_compute(total_flops, "FLOPs"),
                "total_macs_str": self._format_compute(total_macs, "MACs"),
                "params_str": self._format_params(params),
                "flops_per_step_str": self._format_compute(flops_per_step, "FLOPs"),
                "macs_per_step_str": self._format_compute(macs_per_step, "MACs"),
                # Additional breakdowns
                "unet_macs_per_step": m_unet_step,
                "unet_macs_per_step_str": self._format_compute(m_unet_step, "MACs"),
                "vae_macs": m_vae,
                "vae_macs_str": self._format_compute(m_vae, "MACs"),
                "text_encoder_macs": m_text,
                "text_encoder_macs_str": self._format_compute(m_text, "MACs"),
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


def create_profiler(enabled: bool = True) -> ComputeProfiler:
    """Factory function to create a compute profiler.
    
    Args:
        enabled: Whether to enable profiling
        
    Returns:
        ComputeProfiler instance
    """
    return ComputeProfiler(enabled=enabled)
