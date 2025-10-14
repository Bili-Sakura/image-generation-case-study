#!/usr/bin/env python3
"""
Standalone test script for the new thop-based profiler.
Demonstrates the exact style from the user's example.
"""

# pip install diffusers transformers accelerate safetensors thop
import torch
from diffusers import StableDiffusionPipeline
from thop import profile
import torch.nn.functional as F


# --- SDPA FLOPs helper (approx) ---
def sdpa_flops(q, k, v):
    """Calculate FLOPs for SDPA operation."""
    B, H, N, D = q.shape
    mac_qk = B * H * N * N * D
    mac_softmax = B * H * N * N
    mac_av = B * H * N * N * D
    return mac_qk + mac_softmax + mac_av


# Monkeypatch wrapper to count SDPA MACs
_sdpa_macs_counter = {"macs": 0}
_orig_sdpa = None


def _wrap_sdpa_for_macs():
    global _orig_sdpa
    if _orig_sdpa is not None:
        return
    _orig_sdpa = F.scaled_dot_product_attention
    
    def wrapped(q, k, v, *args, **kwargs):
        _sdpa_macs_counter["macs"] += sdpa_flops(q, k, v)
        return _orig_sdpa(q, k, v, *args, **kwargs)
    
    F.scaled_dot_product_attention = wrapped


def _unwrap_sdpa():
    global _orig_sdpa
    if _orig_sdpa is not None:
        F.scaled_dot_product_attention = _orig_sdpa
        _orig_sdpa = None


# --- Build pipeline ---
def test_profiler():
    """Test the profiler with a simple SD pipeline."""
    print("=" * 70)
    print("Testing thop-based Profiler with Stable Diffusion")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load a small model for testing
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading model: {model_id}")
    print("(This may take a while on first run...)\n")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    tokenizer = pipe.tokenizer
    
    def _encode_text(prompt, negative_prompt=""):
        enc = tokenizer([prompt], padding="max_length", truncation=True, 
                       max_length=77, return_tensors="pt").to(device)
        neg = tokenizer([negative_prompt], padding="max_length", truncation=True, 
                       max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            te = pipe.text_encoder(enc.input_ids)[0]
            te_neg = pipe.text_encoder(neg.input_ids)[0]
        return te, te_neg
    
    def measure_unet_macs(height=512, width=512, prompt="a photo of a cat", guidance_scale=7.5):
        pipe.unet.eval()
        with torch.no_grad():
            h8, w8 = height // 8, width // 8
            sample = torch.randn(1, 4, h8, w8, device=device, dtype=pipe.unet.dtype)
            t = torch.tensor([500], device=device, dtype=torch.long)
            cond, uncond = _encode_text(prompt, "")
            
            _wrap_sdpa_for_macs()
            
            def unet_forward_cond(x):
                return pipe.unet(x, t, encoder_hidden_states=cond).sample
            
            macs_cond, _ = profile(unet_forward_cond, inputs=(sample.clone(),), verbose=False)
            
            def unet_forward_uncond(x):
                return pipe.unet(x, t, encoder_hidden_states=uncond).sample
            
            macs_uncond, _ = profile(unet_forward_uncond, inputs=(sample.clone(),), verbose=False)
            
            sdpa_macs = _sdpa_macs_counter["macs"]
            _sdpa_macs_counter["macs"] = 0
            _unwrap_sdpa()
            
            macs_unet_single = (macs_cond + macs_uncond) / 2
            macs_unet_single += sdpa_macs
            per_step = macs_unet_single * (2 if guidance_scale and guidance_scale > 1.0 else 1)
            return int(per_step)
    
    def measure_vae_decode_macs(height=512, width=512):
        pipe.vae.eval()
        with torch.no_grad():
            h8, w8 = height // 8, width // 8
            latents = torch.randn(1, 4, h8, w8, device=device, dtype=pipe.vae.dtype)
            
            def vae_decode(z):
                return pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample
            
            macs, _ = profile(vae_decode, inputs=(latents,), verbose=False)
            return int(macs)
    
    def measure_text_encoder_macs(prompt="a photo of a cat", negative_prompt=""):
        with torch.no_grad():
            enc = tokenizer([prompt], padding="max_length", truncation=True, 
                          max_length=77, return_tensors="pt").to(device)
            neg = tokenizer([negative_prompt], padding="max_length", truncation=True, 
                          max_length=77, return_tensors="pt").to(device)
            
            def run_te(ids):
                return pipe.text_encoder(ids)[0]
            
            macs1, _ = profile(run_te, inputs=(enc.input_ids,), verbose=False)
            macs2, _ = profile(run_te, inputs=(neg.input_ids,), verbose=False)
            return int(macs1 + macs2)
    
    def summarize_macs(height=512, width=512, steps=30, prompt="a photo of a cat", guidance_scale=7.5):
        print("📊 Profiling Components...")
        m_unet_step = measure_unet_macs(height, width, prompt, guidance_scale)
        m_vae = measure_vae_decode_macs(height, width)
        m_text = measure_text_encoder_macs(prompt, "")
        total = m_unet_step * steps + m_vae + m_text
        to_gmac = lambda x: x / 1e9
        return {
            "UNet per-step (MACs)": m_unet_step,
            "UNet per-step (GMACs)": round(to_gmac(m_unet_step), 3),
            "Text encoder once (GMACs)": round(to_gmac(m_text), 3),
            "VAE decode once (GMACs)": round(to_gmac(m_vae), 3),
            f"Total {steps} steps (GMACs)": round(to_gmac(total), 3)
        }
    
    # Run profiling
    result = summarize_macs(height=512, width=512, steps=30, 
                           prompt="a photo of a cat", guidance_scale=7.5)
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 PROFILING RESULTS")
    print("=" * 70)
    for key, value in result.items():
        print(f"  {key:<35} {value}")
    print("=" * 70)
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    try:
        test_profiler()
    except ImportError as e:
        print(f"Error: Missing required library")
        print(f"  {e}")
        print("\nPlease install with:")
        print("  pip install diffusers transformers accelerate safetensors thop")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
