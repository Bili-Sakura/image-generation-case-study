"""
Compute profiler for calculating FLOPs and MACs during inference.
Rewritten to use thop library with SDPA handling.
"""

import torch
from typing import Dict, Optional, Tuple
from diffusers import DiffusionPipeline
import torch.nn.functional as F


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
    
    def measure_unet_macs(
        self, 
        pipe, 
        height: int = 512, 
        width: int = 512, 
        prompt: str = "a photo of a cat", 
        guidance_scale: float = 7.5
    ) -> int:
        """Measure MACs for UNet forward pass.
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
            prompt: Text prompt for conditioning
            guidance_scale: Guidance scale (affects whether we do 1 or 2 passes)
            
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
        
        with torch.no_grad():
            # Latent spatial dims are /8
            h8, w8 = height // 8, width // 8
            
            # Determine latent channels based on model type
            if hasattr(pipe, 'unet'):
                latent_channels = 4  # Standard for SD UNet
            else:
                latent_channels = 16  # Common for transformers
            
            sample = torch.randn(1, latent_channels, h8, w8, device=device, dtype=dtype)
            t = torch.tensor([500], device=device, dtype=torch.long)  # any valid timestep
            
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
            
            # Profile one UNet pass (conditional)
            def unet_forward_cond(x):
                return model(x, t, encoder_hidden_states=cond, return_dict=False)[0]
            
            try:
                macs_cond, _ = self._profile(unet_forward_cond, inputs=(sample.clone(),), verbose=False)
            except:
                macs_cond = 0
            
            # Profile one UNet pass (unconditional)
            def unet_forward_uncond(x):
                return model(x, t, encoder_hidden_states=uncond, return_dict=False)[0]
            
            try:
                macs_uncond, _ = self._profile(unet_forward_uncond, inputs=(sample.clone(),), verbose=False)
            except:
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
            latents = torch.randn(1, 4, h8, w8, device=device, dtype=dtype)
            
            def vae_decode(z):
                scaling_factor = getattr(pipe.vae.config, 'scaling_factor', 0.18215)
                return pipe.vae.decode(z / scaling_factor, return_dict=False)[0]
            
            try:
                macs, _ = self._profile(vae_decode, inputs=(latents,), verbose=False)
                return int(macs)
            except:
                return 0
    
    def measure_text_encoder_macs(
        self, 
        pipe, 
        prompt: str = "a photo of a cat", 
        negative_prompt: str = ""
    ) -> int:
        """Measure MACs for text encoder.
        
        Args:
            pipe: The diffusion pipeline
            prompt: Text prompt
            negative_prompt: Negative text prompt
            
        Returns:
            MACs for text encoding (both prompts)
        """
        if not self._thop_available or not hasattr(pipe, 'text_encoder'):
            return 0
        
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
            
            def run_te(ids):
                return pipe.text_encoder(ids, return_dict=False)[0]
            
            try:
                macs1, _ = self._profile(run_te, inputs=(enc.input_ids,), verbose=False)
            except:
                macs1 = 0
            
            try:
                macs2, _ = self._profile(run_te, inputs=(neg.input_ids,), verbose=False)
            except:
                macs2 = 0
            
            return int(macs1 + macs2)
    
    def summarize_macs(
        self,
        pipe,
        height: int = 512,
        width: int = 512,
        steps: int = 30,
        prompt: str = "a photo of a cat",
        guidance_scale: float = 7.5,
    ) -> Dict[str, any]:
        """Summarize MACs for complete image generation.
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
            steps: Number of inference steps
            prompt: Text prompt
            guidance_scale: Guidance scale
            
        Returns:
            Dictionary with MAC counts and formatted strings
        """
        if not self.enabled or not self._thop_available:
            return {
                "enabled": False,
                "UNet per-step (MACs)": 0,
                "UNet per-step (GMACs)": 0.0,
                "Text encoder once (GMACs)": 0.0,
                "VAE decode once (GMACs)": 0.0,
                f"Total {steps} steps (GMACs)": 0.0,
            }
        
        m_unet_step = self.measure_unet_macs(pipe, height, width, prompt, guidance_scale)
        m_vae = self.measure_vae_decode_macs(pipe, height, width)
        m_text = self.measure_text_encoder_macs(pipe, prompt, "")
        total = m_unet_step * steps + m_vae + m_text
        
        to_gmac = lambda x: x / 1e9
        
        return {
            "UNet per-step (MACs)": m_unet_step,
            "UNet per-step (GMACs)": round(to_gmac(m_unet_step), 3),
            "Text encoder once (GMACs)": round(to_gmac(m_text), 3),
            "VAE decode once (GMACs)": round(to_gmac(m_vae), 3),
            f"Total {steps} steps (GMACs)": round(to_gmac(total), 3),
        }
    
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
            m_unet_step = self.measure_unet_macs(pipe, height, width, "a photo", guidance_scale)
            m_vae = self.measure_vae_decode_macs(pipe, height, width)
            m_text = self.measure_text_encoder_macs(pipe, "a photo", "")
            
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
