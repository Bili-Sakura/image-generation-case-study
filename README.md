# Image Generation Model Profiling

A comprehensive framework for profiling compute costs (MACs/FLOPs) of diffusion-based image generation models. This implementation learns from the diffusers library source code to accurately calculate computational requirements for each model component.

## Overview

This project provides accurate MACs (Multiply-Accumulate Operations) profiling for modern diffusion models by adapting to their diverse architectures. The profiler automatically detects model types and applies appropriate profiling strategies.

## Supported Models

The profiler currently supports 7+ major diffusion model architectures with full component-wise profiling:

- **Stable Diffusion** (1.x, 2.x, XL) - Classic diffusion models
- **Stable Diffusion 3** - Flow-matching transformers
- **FLUX** - Dual-stream attention (partial support)
- **SANA** - High-resolution DiT models
- **PixArt** - Efficient DiT transformers
- **Lumina** - Next-generation DiT models
- **Kandinsky** - Enhanced UNet architecture

## Quick Start

```python
from src.model_manager import get_model_manager
from src.compute_profiler import create_profiler

# Initialize
manager = get_model_manager()
profiler = create_profiler(enabled=True)

# Profile a model
pipe = manager.load_model("stabilityai/stable-diffusion-3-medium-diffusers")
summary = profiler.summarize_macs(
    pipe=pipe,
    height=512,
    width=512,
    steps=30,
    prompt="a photo of a cat",
    guidance_scale=7.5
)

# View results
print(f"Transformer: {summary['UNet per-step (GMACs)']} GMACs/step")
print(f"Text Encoders: {summary['Text encoder once (GMACs)']} GMACs")
print(f"VAE Decoder: {summary['VAE decode once (GMACs)']} GMACs")
print(f"Total: {summary['Total 30 steps (GMACs)']} GMACs")
```

## Results

Computational requirements for 30 inference steps at 512x512:

| Model            | Transformer/UNet (GMACs/step) | Total (TFLOPs) |
| ---------------- | ----------------------------- | -------------- |
| SANA-1.6B        | 14,169.8                      | 442.1          |
| SANA-600M        | 6,314.8                       | 206.5          |
| Lumina-Image-2.0 | 5,113.8                       | 155.0          |
| SD3-medium       | 1,676.2                       | 52.3           |
| PixArt-XL        | 1,350.7                       | 42.5           |
| SD2.1-base       | 1,172.3                       | 36.5           |
| Kandinsky-3      | 935.4                         | 29.4           |

## Key Findings

1. **Transformer dominates compute**: 95%+ of total MACs for all models
2. **SANA models are compute-intensive**: 442 TFLOPs due to 32-channel VAE + large transformer
3. **VAE channels matter**: 32-channel VAE requires 13.5x more compute than 4-channel VAE
4. **Text encoders are cheap**: Typically <2% of total compute

## Architecture

The profiler uses architecture-specific wrappers to handle different forward signatures:

- `UNetWrapper` - Standard diffusion models (SD 1.x, 2.x)
- `SD3TransformerWrapper` - SD3 flow-matching models
- `GenericTransformerWrapper` - DiT models (SANA, PixArt, Qwen, CogView)
- `Lumina2TransformerWrapper` - Lumina-Image models
- `KandinskyUNetWrapper` - Kandinsky enhanced UNet

Model type is automatically detected from class names, and appropriate profiling strategy is applied.

## Project Structure

```
image-generation-case-study/
├── src/
│   ├── compute_profiler.py    # Main profiling implementation
│   ├── model_manager.py        # Model loading and caching
│   ├── config.py               # Model configurations
│   └── inference.py            # Inference utilities
├── docs/
│   └── PROFILING_GUIDE.md      # Detailed documentation
├── final_working_models_test.py # Test script
├── example_profiling_detailed.py # Usage examples
└── README.md                    # This file
```

## Testing

Run the comprehensive test to verify all supported models:

```bash
python final_working_models_test.py
```

Run detailed profiling examples:

```bash
python example_profiling_detailed.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- diffusers
- transformers
- thop (for MACs calculation)

Install profiling dependencies:

```bash
pip install thop
```

## Documentation

- **`docs/PROFILING_GUIDE.md`** - Complete implementation guide, architecture details, and technical insights
- **`docs/RESULTS.md`** - Detailed profiling results and analysis for all tested models

## License

This project follows the same license as the diffusers library.

## Acknowledgments

Implementation based on analysis of the Hugging Face diffusers library source code.
