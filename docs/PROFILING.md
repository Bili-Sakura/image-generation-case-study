# Compute Profiling: FLOPs and MACs Calculation

This document explains how to use the compute profiling features to measure the computational cost of image generation models.

## Overview

The project now includes built-in compute profiling that calculates:

- **FLOPs** (Floating Point Operations): Total number of floating-point operations
- **MACs** (Multiply-Accumulate Operations): Number of multiply-add operations
- **Parameters**: Total number of model parameters
- **Inference Time**: Actual wall-clock time for generation
- **Throughput**: TFLOP/s (tera floating-point operations per second)

This helps you:

- Compare computational efficiency across models
- Understand compute requirements for deployment
- Optimize model selection based on compute budget
- Analyze the impact of different generation parameters

## Installation

The profiling feature uses the `thop` library (PyTorch-OpCounter), which is already included in `environment.yaml`:

```bash
pip install thop
```

Alternatively, you can install a specific version:

```bash
pip install thop==0.1.1-2209072238
```

## Quick Start

### Enable Profiling

Profiling is **enabled by default**. You can control it in three ways:

1. **Global configuration** in `src/config.py`:

```python
ENABLE_COMPUTE_PROFILING = True
```

2. **Per-generation** in function calls:

```python
image, filepath, seed, profiling_data = generate_image(
    model_id="stabilityai/stable-diffusion-2-1-base",
    prompt="A beautiful landscape",
    enable_profiling=True,  # Enable for this generation
)
```

3. **Disable profiling** for faster iteration:

```python
image, filepath, seed, profiling_data = generate_image(
    model_id="stabilityai/stable-diffusion-2-1-base",
    prompt="A beautiful landscape",
    enable_profiling=False,  # Disable profiling
)
```

### Reading Profiling Data

The `generate_image()` function now returns profiling data as the 4th element:

```python
image, filepath, seed, profiling_data = generate_image(...)

if profiling_data and profiling_data.get("enabled"):
    print(f"Parameters: {profiling_data['params_str']}")
    print(f"Total FLOPs: {profiling_data['total_flops_str']}")
    print(f"Total MACs: {profiling_data['total_macs_str']}")
    print(f"Inference Time: {profiling_data['inference_time_str']}")
```

## Profiling Data Structure

The `profiling_data` dictionary contains:

```python
{
    "enabled": True,  # Whether profiling succeeded
    "model_component": "unet",  # Which component was profiled (unet/transformer)

    # Model size
    "total_params": 865909733,
    "params_str": "865.91M",

    # Per-step compute
    "flops_per_step": 123456789000,
    "flops_per_step_str": "123.46 GFLOPs",
    "macs_per_step": 61728394500,
    "macs_per_step_str": "61.73 GMACs",

    # Total compute (flops_per_step × num_inference_steps)
    "total_flops": 6172839450000,
    "total_flops_str": "6.17 TFLOPs",
    "total_macs": 3086419725000,
    "total_macs_str": "3.09 TMACs",

    # Inference performance
    "num_inference_steps": 50,
    "inference_time_seconds": 12.34,
    "inference_time_str": "12.34s",

    # Input/output shapes
    "input_shape": (1, 3, 512, 512),
    "latent_shape": (1, 4, 64, 64),
}
```

## Examples

### Example 1: Basic Profiling

```python
from src.inference import generate_image
from src.model_manager import get_model_manager

manager = get_model_manager()
manager.load_model("stabilityai/stable-diffusion-2-1-base")

image, filepath, seed, profiling_data = generate_image(
    model_id="stabilityai/stable-diffusion-2-1-base",
    prompt="A cat sitting on a windowsill",
    num_inference_steps=50,
    enable_profiling=True,
)

print(f"Model used {profiling_data['total_flops_str']} for generation")
print(f"Inference took {profiling_data['inference_time_str']}")
```

### Example 2: Compare Models

Run the included example script:

```bash
python example_profiling.py
```

This demonstrates:

- Single model profiling
- Multi-model comparison
- Resolution impact analysis

### Example 3: Detailed Component Breakdown

Run the detailed profiling example:

```bash
python example_profiling_detailed.py
```

This shows:

- Component-wise MACs breakdown (UNet, VAE, Text Encoder)
- Percentage contribution of each component
- Compute scaling with different step counts
- Resolution impact analysis with relative costs

Sample output:
```
Component-wise MACs Breakdown:
----------------------------------------------------------------------
  UNet (per step):          339.224 GMACs
  Text Encoder (once):        6.148 GMACs
  VAE Decoder (once):        49.432 GMACs
----------------------------------------------------------------------
  Total (30 steps):        10232.300 GMACs

Percentage Breakdown:
----------------------------------------------------------------------
  UNet:           99.46%
  Text Encoder:    0.06%
  VAE Decoder:     0.48%
```

### Example 4: Batch Profiling

When using batch generation, profiling data is automatically saved to the JSON config file:

```python
from src.inference import generate_images_sequential

results = generate_images_sequential(
    model_ids=[
        "stabilityai/stable-diffusion-2-1-base",
        "PixArt-alpha/PixArt-XL-2-512x512",
    ],
    prompt="A futuristic cityscape",
    num_inference_steps=50,
)
```

Check `outputs/<timestamp>/generation_config.json` for profiling data.

## Output Files

### Console Output

During generation, you'll see profiling information printed:

```
📊 Compute Profile for SD 2.1:
   Parameters: 865.91M
   FLOPs (total): 6.17 TFLOPs
   MACs (total): 3.09 TMACs
   FLOPs/step: 123.46 GFLOPs
   MACs/step: 61.73 GMACs
   Steps: 50

⏱️  Inference time: 12.34s
```

### JSON Config File

Profiling data is automatically saved in `outputs/<timestamp>/generation_config.json`:

```json
{
  "timestamp": "2025-10-14T10:30:00",
  "prompt": "A beautiful landscape",
  "models": [
    {
      "model_id": "stabilityai/stable-diffusion-2-1-base",
      "model_name": "SD 2.1",
      "image_path": "outputs/.../sd21_seed12345.png",
      "compute_profile": {
        "total_params": 865909733,
        "params_str": "865.91M",
        "total_flops": 6172839450000,
        "total_flops_str": "6.17 TFLOPs",
        "total_macs": 3086419725000,
        "total_macs_str": "3.09 TMACs",
        "inference_time_seconds": 12.34,
        "inference_time_str": "12.34s"
      }
    }
  ]
}
```

## Understanding the Metrics

### FLOPs vs MACs

- **FLOPs**: Total floating-point operations (adds, multiplies, etc.)
- **MACs**: Multiply-accumulate operations (one MAC = one multiply + one add)
- Typically: FLOPs ≈ 2 × MACs (since each MAC involves both multiply and add)

### What's Profiled?

The profiler measures the main compute component:

- **UNet** for diffusion models (SD 2.1, SDXL, etc.)
- **Transformer** for transformer-based models (FLUX, SD3, PixArt, etc.)

Note: The profiling excludes VAE, text encoder, and other smaller components, focusing on the main denoising network which dominates compute cost.

### Per-Step vs Total

- **Per-step metrics**: Compute cost for a single denoising step
- **Total metrics**: Cumulative cost over all inference steps
- Formula: `total = per_step × num_inference_steps`

## Performance Considerations

### When to Disable Profiling

Profiling adds minimal overhead, but you may want to disable it:

- During rapid prototyping/testing
- When running large batch jobs where compute info isn't needed
- If `calflops` is not installed

### Profiling Limitations

Some models may not support profiling due to:

- Custom layers not recognized by calflops
- Complex model architectures
- Dynamically constructed computation graphs

In these cases, the profiler will gracefully fail and return empty profiling data without breaking generation.

## Troubleshooting

### Profiling Not Working

If profiling is disabled, check:

1. Is `calflops` installed?

   ```bash
   pip install calflops
   ```

2. Is profiling enabled in config?

   ```python
   # src/config.py
   ENABLE_COMPUTE_PROFILING = True
   ```

3. Check console warnings:
   ```
   ⚠️  thop not installed. Install with: pip install thop
   ```

### Unexpected Results

If FLOPs/MACs seem incorrect:

- Different model architectures have vastly different compute costs
- Resolution significantly impacts compute (4x resolution ≈ 16x compute)
- Number of steps linearly scales total compute
- Check that the correct component (UNet/Transformer) is being profiled

## Advanced Usage

### Custom Profiling

You can use the profiler directly:

```python
from src.compute_profiler import create_profiler

profiler = create_profiler(enabled=True)
profiling_data = profiler.profile_pipeline(
    pipe=pipeline,
    input_shape=(1, 3, 512, 512),
    num_inference_steps=50,
    model_id="my-model",
)
```

### Integrating into Your Code

If you're building custom generation workflows:

```python
from src.compute_profiler import create_profiler
from src.model_manager import get_model_manager

manager = get_model_manager()
pipe = manager.load_model("model-id")

profiler = create_profiler(enabled=True)
profiling_data = profiler.profile_pipeline(
    pipe=pipe,
    input_shape=(1, 3, height, width),
    num_inference_steps=steps,
    model_id="model-id",
)

# Use profiling_data...
```

## Technical Details

### thop Library

The profiler now uses `thop` (PyTorch-OpCounter) which provides:

- Accurate MAC counting for conv and linear layers
- Lightweight and fast profiling
- Better compatibility with custom modules

### SDPA Handling

The profiler includes special handling for Scaled Dot Product Attention (SDPA):

- Automatically counts attention MACs: `QK^T + softmax + attention @ V`
- Formula: `MACs = 2 * B * heads * N^2 * head_dim + B * heads * N^2`
- Integrated via monkeypatch during profiling passes

### Component Breakdown

The new profiler measures three components separately:

1. **UNet/Transformer**: Main denoising network (per step)
2. **Text Encoder**: CLIP text encoding (once per generation)
3. **VAE Decoder**: Latent to image decoding (once per generation)

Total compute: `UNet × steps + Text Encoder + VAE Decoder`

## References

- [thop library (PyTorch-OpCounter)](https://github.com/Lyken17/pytorch-OpCounter)
- [Understanding FLOPs in Deep Learning](https://medium.com/swlh/understanding-flops-in-deep-learning-6f5eae2f4f0b)
- [Efficient Diffusion Models](https://arxiv.org/abs/2209.00796)
- [Attention Mechanisms and Compute](https://arxiv.org/abs/1706.03762)

## Future Enhancements

Planned improvements:

- [ ] Profile VAE and text encoder separately
- [ ] GPU memory profiling
- [ ] Comparative analysis dashboard
- [ ] Export profiling data to CSV/charts
- [ ] Profile attention mechanism breakdown
