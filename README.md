# README

This repo hold codes for case study of existing open-sourced diffusion models' capability in text-to-image generation and image editing.

## Model List

### Open-Source Text-to-Image Models:

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

### Closed-Source API Services:

- **OpenAI DALL-E**: DALL-E 2 & DALL-E 3 (with quality and style controls)
- **Google Imagen**: Vertex AI Imagen (high-quality photorealistic generation)
- **Bytedance Cloud**: Volcano Engine text-to-image API
- **Kling AI**: Kling image generation models

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Bili-Sakura/image-generation-case-study.git
cd image-generation-case-study
```

2. Install dependencies:

```bash
# Install base dependencies for open-source models
pip install -r requirements.txt

# Optional: Install API dependencies for closed-source models
pip install -r requirements_api.txt
```

3. (Optional) Configure API keys for closed-source models:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

Then set the environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_PROJECT_ID="your_google_project_id"
export BYTEDANCE_API_KEY="your_bytedance_api_key"
export KLING_API_KEY="your_kling_api_key"
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
- 🔓 Open-Source Models: 14 local diffusion models
- 🔒 Closed-Source APIs: OpenAI, Google, Bytedance, Kling
- 🔄 API Comparison: Batch generation across multiple API providers
- ⚙️ Configurable parameters: Steps, guidance, size, seed
- 🖼️ Gallery view: See all results with model labels
- 💾 Auto-save: Images saved to `/outputs/{timestamp}/` with generation config JSON
- 📊 Memory efficient: Sequential generation
- 🚀 Multi-GPU support: Automatic device mapping for utilizing multiple GPUs

**Developer Mode:**

```bash
python run.py --dev
```

Allows manual model selection and one-time batch loading (port 7861).

**Batch Mode:**

```bash
python run.py --batch
```

Tests all models sequentially without pre-loading. Each model is loaded, used for inference, then unloaded to minimize memory usage (port 7862).

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

#### Option 3: Closed-Source API Usage

Use the API clients programmatically:

```python
from src.api_clients import get_api_client

# OpenAI DALL-E
client = get_api_client("openai")
image, error = client.generate(
    prompt="A serene Japanese garden with a red bridge",
    width=1024,
    height=1024,
    model="dall-e-3",
    quality="hd",
    style="vivid"
)

if image:
    image.save("output.png")
else:
    print(f"Error: {error}")
```

See `example_api_generate.py` for more examples including batch comparisons across multiple APIs.

## Application Modes

This application provides three different modes for different use cases:

### 1. User Mode (Default - Port 7860)

**Launch:**

```bash
python run.py
```

**Features:**

- Pre-loads default models on startup for fast generation
- Allows model selection from available models
- Supports both open-source and closed-source (API) models
- Ideal for regular usage and quick comparisons

### 2. Developer Mode (Port 7861)

**Launch:**

```bash
python run.py --dev
```

**Features:**

- Manual model selection with checkboxes
- One-time batch loading of selected models
- All selected models stay loaded in memory
- Accumulative loading (can add more models without unloading)
- Fast sequential inference with pre-loaded models
- Ideal for development and testing specific model combinations

### 3. Batch Mode (Port 7862)

**Launch:**

```bash
python run.py --batch
```

**Features:**

- **No pre-loading** - starts with empty memory
- **No model selection** - automatically tests all available models
- **Sequential processing**: Load → Inference → Unload (one by one)
- Minimizes memory usage (only one model loaded at a time)
- Ideal for comprehensive benchmarking and testing
- Perfect for limited VRAM scenarios

**Comparison:**

| Feature          | User Mode | Developer Mode | Batch Mode   |
| ---------------- | --------- | -------------- | ------------ |
| Pre-loading      | Yes       | Manual         | No           |
| Model Selection  | Yes       | Yes            | No (All)     |
| Memory Usage     | Medium    | High           | Low          |
| Generation Speed | Fast      | Fastest        | Slow         |
| Ideal For        | General   | Development    | Benchmarking |

## Project Structure

```
image-generation-case-study/
├── src/                          # Application source code
│   ├── app.py                    # Main Gradio UI (user mode)
│   ├── app_dev.py                # Developer mode UI
│   ├── app_batch.py              # Batch mode UI (all models)
│   ├── config.py                 # Model configurations
│   ├── model_manager.py          # Model loading & caching
│   ├── inference.py              # Generation logic
│   ├── api_clients.py            # Closed-source API clients
│   ├── closed_source_widget.py   # Gradio widget for APIs
│   ├── utils.py                  # Utility functions
│   └── README.md                 # Detailed documentation
├── outputs/                      # Generated images (organized by timestamp)
├── libs/                         # Reference libraries
│   ├── diffusers/
│   ├── transformers/
│   └── ...
├── requirements.txt              # Python dependencies (base)
├── requirements_api.txt          # API dependencies (optional)
├── .env.example                  # Environment variable template
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

### Closed-Source API Models

| Provider        | Models Available   | Max Resolution | Special Features          |
| --------------- | ------------------ | -------------- | ------------------------- |
| OpenAI DALL-E   | DALL-E 2, DALL-E 3 | 1792x1792      | Quality & style controls  |
| Google Imagen   | Imagen v5          | 1536x1536      | Photorealistic generation |
| Bytedance Cloud | Text2Img v1        | 2048x2048      | Fast API response         |
| Kling AI        | Kling v1, v1 Pro   | 2048x2048      | High-quality generation   |

**API Features:**

- 🔑 Secure API key management via environment variables
- 🎛️ Provider-specific controls (quality, style for DALL-E)
- 🔄 Batch comparison across multiple APIs
- 💰 Pay-per-use model (requires API credits)
- ⚡ No GPU required (cloud-based generation)

### Generation Parameters

- **Inference Steps**: 10-100 (default: 50)
- **Guidance Scale**: 1.0-20.0 (default: 7.5)
- **Image Sizes**: 512px to 1280px (multiple presets)
- **Seed Control**: Fixed or random (-1)
- **Negative Prompts**: Supported on compatible models

### Multi-GPU Support

The application automatically detects and utilizes multiple GPUs when available:

- **Automatic Device Mapping**: When 2+ GPUs are detected, models are automatically distributed across GPUs using `device_map="auto"`
- **Transparent Operation**: No configuration needed - just run the application
- **GPU Information**: The UI displays all available GPUs and their memory
- **Single GPU Fallback**: Automatically falls back to single GPU mode if only one GPU is available

The system will print GPU information on startup:

```
Found 2 GPU(s):
  GPU 0: NVIDIA A100-SXM4-40GB (40.0 GB)
  GPU 1: NVIDIA A100-SXM4-40GB (40.0 GB)
Multi-GPU mode enabled: 2 GPUs detected
```

### Output Organization

Images are automatically organized by timestamp:

```
/outputs/
├── 20251009_140530/
│   ├── SD2.1_20251009_140530_seed12345.png
│   ├── SDXL_20251009_140531_seed12345.png
│   ├── FLUX.1-dev_20251009_140532_seed12345.png
│   └── generation_config.json
├── 20251009_141030/
│   └── ...
└── ...
```

Each generation batch includes:

- Multiple PNG image files (one per model)
- Single JSON config file with all generation parameters and model information

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
