# Quick Profiler Guide

## Installation

```bash
pip install thop
```

## Quick Start

### 1. Automatic Profiling (Recommended)

```python
from src.inference import generate_image

image, filepath, seed, profiling_data = generate_image(
    model_id="stabilityai/stable-diffusion-2-1-base",
    prompt="A beautiful landscape",
    enable_profiling=True,  # Enable profiling
)

# Access results
if profiling_data and profiling_data.get("enabled"):
    print(f"Total MACs: {profiling_data['total_macs_str']}")
    print(f"UNet per step: {profiling_data['macs_per_step_str']}")
    print(f"Parameters: {profiling_data['params_str']}")
```

### 2. Manual Profiling

```python
from src.compute_profiler import create_profiler
from src.model_manager import get_model_manager

profiler = create_profiler(enabled=True)
manager = get_model_manager()
pipe = manager.load_model("stabilityai/stable-diffusion-2-1-base")

# Get detailed breakdown
summary = profiler.summarize_macs(
    pipe=pipe,
    height=512,
    width=512,
    steps=30,
    prompt="a photo of a cat",
    guidance_scale=7.5,
)

print(summary)
```

### 3. Component-wise Analysis

```python
# Measure individual components
unet_macs = profiler.measure_unet_macs(pipe, 512, 512, "prompt", 7.5)
vae_macs = profiler.measure_vae_decode_macs(pipe, 512, 512)
text_macs = profiler.measure_text_encoder_macs(pipe, "prompt", "")

print(f"UNet: {unet_macs / 1e9:.2f} GMACs")
print(f"VAE: {vae_macs / 1e9:.2f} GMACs")
print(f"Text: {text_macs / 1e9:.2f} GMACs")
```

## Run Examples

```bash
# Basic profiling
python example_profiling.py

# Detailed analysis with breakdowns
python example_profiling_detailed.py

# Standalone test
python test_profiler_standalone.py
```

## Output Interpretation

### Typical Values (SD 2.1, 512×512, 30 steps)

```
Component-wise Breakdown:
  UNet (per step):        ~339 GMACs
  Text Encoder (once):     ~6 GMACs
  VAE Decoder (once):     ~49 GMACs
  Total (30 steps):    ~10,232 GMACs

Percentage:
  UNet:           99.46%  (dominates compute)
  Text Encoder:    0.06%
  VAE Decoder:     0.48%
```

### What affects compute?

1. **Resolution** - Scales quadratically
   - 512×512 → 768×768 = ~2.25× compute
   - Due to spatial attention: O(N²) where N = H×W

2. **Steps** - Scales linearly
   - 30 steps → 50 steps = ~1.67× compute
   - UNet dominates, so step count has major impact

3. **Guidance Scale** - Binary effect
   - CFG > 1.0 = 2× UNet forward passes
   - CFG = 1.0 = 1× UNet forward pass

## API Reference

### `ComputeProfiler` Methods

```python
profiler = create_profiler(enabled=True)

# Measure UNet/Transformer
macs = profiler.measure_unet_macs(pipe, height, width, prompt, guidance_scale)

# Measure VAE decoder
macs = profiler.measure_vae_decode_macs(pipe, height, width)

# Measure text encoder
macs = profiler.measure_text_encoder_macs(pipe, prompt, negative_prompt)

# Get complete summary
summary = profiler.summarize_macs(pipe, height, width, steps, prompt, guidance_scale)

# Full pipeline profiling
data = profiler.profile_pipeline(pipe, input_shape, num_steps, model_id, guidance_scale)
```

### Profiling Data Structure

```python
{
    "enabled": True,
    "model_component": "unet",  # or "transformer"
    
    # Raw values
    "total_macs": 10232300000000,
    "total_flops": 20464600000000,
    "macs_per_step": 341077000000,
    "flops_per_step": 682154000000,
    "total_params": 865909733,
    
    # Formatted strings
    "total_macs_str": "10.23 TMACs",
    "total_flops_str": "20.46 TFLOPs",
    "macs_per_step_str": "341.08 GMACs",
    "params_str": "865.91M",
    
    # Breakdowns (if available)
    "unet_macs_per_step": 339224000000,
    "vae_macs": 49432000000,
    "text_encoder_macs": 6148000000,
}
```

## Troubleshooting

### Issue: "thop not installed"
```bash
pip install thop
# or
pip install thop==0.1.1-2209072238
```

### Issue: Profiling returns all zeros
- Check if `thop` is properly installed
- Ensure model is loaded correctly
- Verify device placement (CPU vs GPU)

### Issue: Numbers seem too high/low
- Different models have vastly different compute
- Resolution has quadratic impact
- Check if CFG is enabled (2× compute if > 1.0)

## Technical Notes

### SDPA (Scaled Dot Product Attention)

The profiler includes special handling for attention:

```python
MACs = B × H × N² × D  (QK^T)
     + B × H × N²       (softmax)
     + B × H × N² × D   (attn @ V)
```

This is automatically counted via monkeypatch during profiling.

### FLOPs vs MACs

- **MAC** = 1 multiply + 1 add = 2 FLOPs
- **FLOPs** ≈ 2 × MACs (typically)

We report both for completeness.

## See Also

- **Full Documentation:** `docs/PROFILING.md`
- **Detailed Examples:** `example_profiling_detailed.py`
- **Standalone Test:** `test_profiler_standalone.py`
- **Summary:** `PROFILER_REWRITE_SUMMARY.md`
