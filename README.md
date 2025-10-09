# README

This repo hold codes for case study of existing open-sourced diffusion models' capability in text-to-image generation and image editing.

## Model List

Text-to-Image:

- [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [zai-org/CogView3-Plus-3B](https://huggingface.co/zai-org/CogView3-Plus-3B)
- [PixArt-alpha/PixArt-XL-2-512x512](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512)
- [PixArt-alpha/PixArt-Sigma-XL-2-512-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS)
- [Alpha-VLLM/Lumina-Next-SFT-diffusers](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)
- [Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers)
- [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Efficient-Large-Model/Sana_600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- [thu-ml/unidiffuser-v1](https://huggingface.co/thu-ml/unidiffuser-v1)
- [stabilityai/stable-cascade](https://huggingface.co/stabilityai/stable-cascade)
- [zai-org/CogView4-6B](https://huggingface.co/zai-org/CogView4-6B)

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Bili-Sakura/image-generation-case-study.git
cd image-generation-case-study
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Gradio Web UI (Recommended)

Launch the interactive web interface:

```bash
python run.py
```

Or run directly:

```bash
python -m src.app
```

This will:

- Open a web browser at `http://localhost:7860`
- Pre-load default models (SD 2.1, SDXL, Stable Cascade, SD3)
- Provide an intuitive UI for text-to-image generation

**Features:**

- 🎨 Multi-model comparison: Generate with multiple models simultaneously
- ⚙️ Configurable parameters: Steps, guidance, size, seed
- 🖼️ Gallery view: See all results with model labels
- 💾 Auto-save: Images saved to `outputs/{model_name}/`
- 📊 Memory efficient: Sequential generation

**Developer Mode:**

```bash
python run.py --dev
```

Allows manual model selection and one-time batch loading (port 7861).

#### Option 2: Python API

Use the modules programmatically:

```python
from diffusers import DiffusionPipeline

# Load a pre-trained diffusion pipeline (e.g., Stable Diffusion 2-1)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

# Generate an image from a text prompt
prompt = "A fantasy landscape with mountains and rivers"
image = pipe(prompt).images[0]

# Save the generated image
image.save("generated_image.png")
```

Or use the built-in model manager:

```python
from src.model_manager import get_model_manager
from src.inference import generate_image

# Load model
manager = get_model_manager()
manager.load_model("stabilityai/stable-diffusion-2-1")

# Generate
image, filepath, seed = generate_image(
    model_id="stabilityai/stable-diffusion-2-1",
    prompt="A fantasy landscape with mountains and rivers",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)
```

## Project Structure

```
image-generation-case-study/
├── src/                          # Application source code
│   ├── app.py                    # Main Gradio UI (user mode)
│   ├── app_dev.py                # Developer mode UI
│   ├── config.py                 # Model configurations
│   ├── model_manager.py          # Model loading & caching
│   ├── inference.py              # Generation logic
│   ├── utils.py                  # Utility functions
│   └── README.md                 # Detailed documentation
├── outputs/                      # Generated images (organized by model)
├── libs/                         # Reference libraries
│   ├── diffusers/
│   ├── transformers/
│   └── ...
├── requirements.txt              # Python dependencies
├── run.py                        # Launcher script
└── README.md                     # This file
```

## Features

### Supported Models (14 Total)

| Model                | Size   | VRAM   | Resolution | Notes                 |
| -------------------- | ------ | ------ | ---------- | --------------------- |
| Stable Diffusion 2.1 | Base   | ~4 GB  | 768x768    | Classic, reliable     |
| Stable Diffusion XL  | XL     | ~7 GB  | 1024x1024  | Higher quality        |
| CogView3 Plus 3B     | 3B     | ~6 GB  | 1024x1024  | Multilingual          |
| PixArt-XL 2          | XL     | ~4 GB  | 512x512    | Fast generation       |
| PixArt-Sigma XL 2    | XL     | ~4 GB  | 512x512    | Improved PixArt       |
| Lumina-Next SFT      | Large  | ~8 GB  | 1024x1024  | Advanced architecture |
| HunyuanDiT v1.2      | Large  | ~10 GB | 1024x1024  | Chinese + English     |
| Stable Diffusion 3   | Medium | ~9 GB  | 1024x1024  | Latest SD3            |
| FLUX.1 Dev           | XL     | ~16 GB | 1024x1024  | State-of-the-art      |
| Sana 600M            | Small  | ~3 GB  | 512x512    | Lightweight           |
| Qwen Image           | Large  | ~8 GB  | 1024x1024  | Multimodal            |
| UniDiffuser v1       | Base   | ~5 GB  | 512x512    | Unified model         |
| Stable Cascade       | Large  | ~10 GB | 1024x1024  | Multi-stage           |
| CogView4 6B          | 6B     | ~11 GB | 1024x1024  | Latest CogView        |

**Default models** (pre-loaded on startup):

- Stable Diffusion 2.1
- Stable Diffusion XL
- Stable Cascade
- Stable Diffusion 3

### Generation Parameters

- **Inference Steps**: 10-100 (default: 50)
- **Guidance Scale**: 1.0-20.0 (default: 7.5)
- **Image Sizes**: 512px to 1280px (multiple presets)
- **Seed Control**: Fixed or random (-1)
- **Negative Prompts**: Supported on compatible models

### Output Organization

Images are automatically organized:

```
outputs/
├── stabilityai-stable-diffusion-2-1/
│   ├── 20251009_140530_seed12345.png
│   └── 20251009_140530_seed12345.txt
├── stabilityai-stable-diffusion-xl-base-1.0/
│   └── ...
└── ...
```

Each generation includes:

- PNG image file
- TXT metadata (model, prompt, seed, timestamp)

## Citation

If you find this repository useful, please cite it as:

```bibtex
@misc{bili_sakura_image_generation_case_study,
  author       = {Bili-Sakura},
  title        = {Image Generation Case Study},
  year         = {2025},
  howpublished = {\url{https://github.com/Bili-Sakura/image-generation-case-study}},
  note         = {Accessed: 2025-10-08}
}
```
