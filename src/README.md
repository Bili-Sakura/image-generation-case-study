# Text-to-Image Generation Application

This directory contains the implementation of a multi-model text-to-image generation tool using Gradio and Diffusers.

## Structure

```
src/
├── __init__.py           # Package initialization
├── config.py             # Model configurations and constants
├── model_manager.py      # Model loading and memory management
├── inference.py          # Image generation logic
├── utils.py              # Utility functions
├── app.py                # Main user-facing Gradio UI
└── app_dev.py            # Developer mode with manual model selection
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

**User Mode** (recommended for most users):

```bash
python -m src.app
```

This will:

- Pre-load default models (SD 2.1, SDXL, Stable Cascade, SD3)
- Launch Gradio UI on http://localhost:7860
- Allow sequential generation with selected models

**Developer Mode**:

```bash
python -m src.app_dev
```

This mode:

- Allows manual model selection before loading
- Loads all selected models at once
- Launches on http://localhost:7861
- Better for testing specific models

## Features

### User Interface

- **Single-page design** with collapsible sections
- **Model selection** via checkboxes
- **Configurable parameters**:
  - Number of inference steps (10-100)
  - Guidance scale (1.0-20.0)
  - Image size presets
  - Random seed control
- **Gallery view** showing all generated images with model labels
- **Automatic saving** to `outputs/{model_name}/` directories

### Memory Management

- **Sequential generation**: Processes one model at a time
- **Model caching**: Loaded models stay in memory
- **No offloading**: Direct GPU inference for speed
- **Manual loading**: Option to pre-load specific models

### Supported Models

The application supports 14 state-of-the-art diffusion models:

1. **Stable Diffusion 2.1** - Classic SD model (~4 GB VRAM)
2. **Stable Diffusion XL** - Higher quality SDXL (~7 GB VRAM)
3. **CogView3 Plus 3B** - Efficient multilingual model (~6 GB VRAM)
4. **PixArt-XL 2** - Fast high-resolution generation (~6 GB VRAM)
5. **PixArt-Sigma XL 2** - Improved PixArt variant (~6 GB VRAM)
6. **Lumina-Next SFT** - Advanced architecture (~8 GB VRAM)
7. **HunyuanDiT v1.2** - Tencent's powerful model (~10 GB VRAM)
8. **Stable Diffusion 3 Medium** - Latest SD3 (~9 GB VRAM)
9. **FLUX.1 Dev** - State-of-the-art quality (~16 GB VRAM)
10. **Sana 600M** - Lightweight efficient model (~3 GB VRAM)
11. **Qwen Image** - Qwen's image generation (~8 GB VRAM)
12. **UniDiffuser v1** - Unified multimodal model (~5 GB VRAM)
13. **Stable Cascade** - Multi-stage generation (~10 GB VRAM)
14. **CogView4 6B** - Latest CogView model (~11 GB VRAM)

## Usage Examples

### Basic Generation

1. Enter your prompt: "A serene landscape with mountains"
2. Select models (default: SD 2.1, SDXL, Cascade, SD3)
3. Click "Generate Images"
4. View results in the gallery

### Advanced Configuration

1. Open "Generation Settings" accordion
2. Adjust parameters:
   - Steps: 50 (balanced quality/speed)
   - Guidance: 7.5 (standard adherence)
   - Size: 1024x1024
   - Seed: Use -1 for random, or specific number for reproducibility
3. Generate and compare

### Developer Workflow

1. Launch `app_dev.py`
2. Go to "Model Setup" tab
3. Select specific models to test
4. Click "Load Selected Models"
5. Switch to "Generate" tab
6. Test with different prompts

## Output Organization

Generated images are saved to:

```
outputs/
├── stabilityai-stable-diffusion-2-1/
│   ├── 20251009_143022_seed12345.png
│   └── 20251009_143022_seed12345.txt
├── stabilityai-stable-diffusion-xl-base-1.0/
│   ├── 20251009_143045_seed12345.png
│   └── 20251009_143045_seed12345.txt
└── ...
```

Each image comes with a `.txt` metadata file containing:

- Model ID
- Prompt
- Seed
- Timestamp

## Configuration

### Unified Configuration Approach

All models use the same unified `DEFAULT_CONFIG` for generation parameters. This ensures consistent behavior across all models and simplifies configuration management.

### Default Models

Edit `DEFAULT_MODELS` in `config.py` to change which models are pre-loaded on startup:

```python
DEFAULT_MODELS = [
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-cascade",
    "stabilityai/stable-diffusion-3-medium-diffusers",
]
```

### Generation Defaults

Edit `DEFAULT_CONFIG` in `config.py` to change the default generation parameters used by all models:

```python
DEFAULT_CONFIG = {
    "num_inference_steps": 50,
    "guidance_scale": 1.0,  # 1.0 = no CFG amplification (2x faster)
    "width": 512,
    "height": 512,
    "seed": -1,
    "negative_prompt": "",
    "scheduler": "EulerDiscreteScheduler",  # Unified scheduler for all models
}
```

**Important Notes:**

- All models use these unified defaults unless overridden by user input in the UI or API
- **Guidance Scale = 1.0**: Provides ~2x faster inference (single forward pass) while still using the text prompt. For stronger prompt adherence, increase to 7-9 (slower, uses classifier-free guidance)
- **Unified Scheduler**: EulerDiscreteScheduler is applied to all models for consistent sampling behavior

### Image Size Presets

Add/modify in `IMAGE_SIZE_PRESETS` in `config.py`:

```python
IMAGE_SIZE_PRESETS = {
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    # Add custom sizes...
}
```

## API Usage

You can also use the modules programmatically:

```python
from src.model_manager import get_model_manager
from src.inference import generate_image

# Initialize manager
manager = get_model_manager()
manager.load_model("stabilityai/stable-diffusion-2-1")

# Generate image
image, filepath, seed = generate_image(
    model_id="stabilityai/stable-diffusion-2-1",
    prompt="A beautiful sunset",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
)

print(f"Image saved to: {filepath}")
```

## Performance Tips

1. **VRAM Management**: Start with fewer models if you have limited VRAM
2. **Speed Optimization**: Use fewer steps (20-30) for faster generation
3. **Quality**: Use 50+ steps and guidance 7-9 for best results
4. **Comparison**: Use the same seed across models for fair comparison
5. **Batch Processing**: Select multiple models to compare styles

## Troubleshooting

### Out of Memory Errors

- Reduce number of selected models
- Use smaller image sizes
- Close other GPU applications

### Slow Generation

- Decrease inference steps
- Use smaller models (Sana, SD 2.1)
- Check GPU utilization

### Model Loading Fails

- Check internet connection (models download from HuggingFace)
- Verify HuggingFace credentials for gated models
- Check available disk space for model cache

## Contributing

To add a new model:

1. Add model configuration to `MODELS` in `config.py`:

```python
"org/model-name": {
    "name": "Model Full Name",
    "short_name": "Short",
    "requires_safety_checker": False,
    "pipeline_class": "PipelineClassName",
}
```

2. The model will automatically appear in the UI and use the unified `DEFAULT_CONFIG` for generation parameters

## License

See main repository LICENSE file.
