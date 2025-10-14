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
- [kandinsky-community/kandinsky-3](https://huggingface.co/kandinsky-community/kandinsky-3)
- [HiDream-ai/HiDream-I1-Dev](https://huggingface.co/HiDream-ai/HiDream-I1-Dev)
- [Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)

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

Launch the interactive web interface by choosing a mode:

**Developer Mode (Recommended for regular use):**

```bash
python run.py --dev
```

**Benchmark Mode (For comprehensive evaluation):**

```bash
python run.py --bench
```

**Features:**

- 🎨 Multi-model comparison: Generate with multiple models simultaneously
- 🔓 Open-Source Models: 18 local diffusion models
- 🔒 Closed-Source APIs: OpenAI, Google, Bytedance, Kling (in developer mode)
- 🔄 API Comparison: Batch generation across multiple API providers (in developer mode)
- ⚙️ Configurable parameters: Steps, guidance, size, seed
- 🖼️ Gallery view: See all results with model labels
- 💾 Auto-save: Images saved to `/outputs/{timestamp}/` with generation config JSON
- 📊 Memory efficient: Sequential generation
- 🚀 Multi-GPU support: Automatic device mapping for utilizing multiple GPUs
- 🔬 **Compute Profiling**: FLOPs and MACs calculation to measure computational cost

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

# Generate with compute profiling
image, filepath, seed, profiling_data = generate_image(
    model_id="stabilityai/stable-diffusion-2-1",
    prompt="A fantasy landscape with mountains and rivers",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    enable_profiling=True  # Enable FLOPs/MACs calculation
)

# View compute statistics
if profiling_data and profiling_data.get("enabled"):
    print(f"Parameters: {profiling_data['params_str']}")
    print(f"Total FLOPs: {profiling_data['total_flops_str']}")
    print(f"Inference Time: {profiling_data['inference_time_str']}")
```

See `docs/PROFILING.md` for detailed documentation on compute profiling.

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

This application provides two different modes for different use cases:

### 1. Developer Mode (Port 7861)

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

### 2. Benchmark Mode (Port 7862)

**Launch:**

```bash
python run.py --bench
```

**Features:**

- **No pre-loading** - starts with clean memory state
- **No model selection** - systematically evaluates all available models
- **Sequential workflow**: Load → Generate → Unload (per model)
- Minimizes memory footprint (only one model loaded at a time)
- Ensures fair comparison with fresh start for each model
- Ideal for comprehensive evaluation and performance benchmarking
- Perfect for limited VRAM scenarios

**Comparison:**

| Feature          | Developer Mode | Benchmark Mode |
| ---------------- | -------------- | -------------- |
| Pre-loading      | Manual         | No             |
| Model Selection  | Yes            | No (All)       |
| Memory Usage     | High           | Low            |
| Generation Speed | Fastest        | Slower         |
| Use Case         | Development    | Evaluation     |
| Port             | 7861           | 7862           |

## Project Structure

```
image-generation-case-study/
├── src/                          # Application source code
│   ├── app_dev.py                # Developer mode UI (port 7861)
│   ├── app_bench.py              # Benchmark mode UI (port 7862)
│   ├── config.py                 # Model configurations
│   ├── model_manager.py          # Model loading & caching
│   ├── inference.py              # Generation logic
│   ├── compute_profiler.py       # FLOPs/MACs profiling utilities
│   ├── api_clients.py            # Closed-source API clients
│   ├── closed_source_widget.py   # Gradio widget for APIs
│   ├── utils.py                  # Utility functions
│   └── README.md                 # Detailed documentation
├── docs/                         # Documentation
│   └── PROFILING.md              # Compute profiling guide
├── outputs/                      # Generated images (organized by timestamp)
├── libs/                         # Reference libraries
│   ├── diffusers/
│   ├── transformers/
│   └── ...
├── requirements.txt              # Python dependencies (base)
├── requirements_api.txt          # API dependencies (optional)
├── example_generate.py           # Example usage
├── example_profiling.py          # Profiling examples
├── .env.example                  # Environment variable template
├── run.py                        # Launcher script (requires --dev or --bench)
└── README.md                     # This file
```

## Features

### Supported Models (18 Total)

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
| Kandinsky 3          | Large  | ~6 GB  | 1024x1024  | Russian model         |
| HiDream I1 Dev       | Large  | ~12 GB | 1024x1024  | High-quality results  |
| SANA 1.5 1.6B        | 1.6B   | ~4 GB  | 1024x1024  | Efficient generation  |
| Lumina Image 2.0     | Large  | ~8 GB  | 1024x1024  | Advanced Lumina       |

**Note:** In developer mode, you can select which models to load. In benchmark mode, all models are evaluated automatically with consistent parameters.

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

### Compute Profiling (NEW!)

Measure the computational cost of different models during inference:

- **FLOPs** (Floating Point Operations): Total compute operations
- **MACs** (Multiply-Accumulate Operations): Number of multiply-add operations
- **Parameters**: Model size (number of parameters)
- **Inference Time**: Actual wall-clock generation time
- **Throughput**: TFLOP/s (computational efficiency)

**Quick Example:**

```bash
# Run profiling examples
python example_profiling.py
```

**Features:**

- ✅ Automatic profiling during generation (enabled by default)
- 📊 Profiling data saved in JSON config files
- 🔍 Compare compute cost across models
- ⚡ Minimal overhead (~1-2% slowdown)
- 📈 Analyze impact of resolution and steps on compute

**Learn More:** See [`docs/PROFILING.md`](docs/PROFILING.md) for detailed documentation and advanced usage.

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
- Single JSON config file with all generation parameters, model information, and compute profiling data (FLOPs, MACs, inference time)

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
