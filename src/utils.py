"""
Utility functions for image processing and file handling.
"""

from datetime import datetime
from pathlib import Path
import torch
import math
from PIL import Image
from typing import List, Optional
from diffusers.utils import make_image_grid
from src.config import MODELS


def get_model_output_dir(model_id: str, base_dir: str = "outputs") -> Path:
    """Get or create output directory for a specific model."""
    output_path = Path(base_dir) / model_id.replace("/", "-")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_image(image: Image.Image, model_id: str, seed: int, prompt: str, base_dir: str = "outputs") -> str:
    """Save generated image with metadata."""
    output_dir = get_model_output_dir(model_id, base_dir)
    model_short_name = MODELS.get(model_id, {}).get("short_name", model_id)
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_short_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filepath = output_dir / f"{safe_name}_{timestamp}_seed{seed}.png"
    image.save(filepath)
    
    # Save metadata
    filepath.with_suffix(".txt").write_text(
        f"Model: {model_id}\nPrompt: {prompt}\nSeed: {seed}\nTimestamp: {timestamp}\n"
    )
    return str(filepath)


def set_seed(seed: int) -> int:
    """Set random seed for reproducibility. Returns the seed used."""
    import random
    seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_device() -> str:
    """Get the appropriate device (cuda/cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_info() -> str:
    """Get information about available GPUs."""
    if not torch.cuda.is_available():
        return "No CUDA GPUs available"
    
    gpu_count = torch.cuda.device_count()
    info = [f"Found {gpu_count} GPU(s):"]
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        info.append(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")
    
    return "\n".join(info)


def estimate_memory_usage(model_id: str) -> str:
    """Estimate VRAM usage for a model (rough estimates)."""
    # These are rough estimates in GB
    memory_estimates = {
        "stabilityai/stable-diffusion-2-1": "~4 GB",
        "stabilityai/stable-diffusion-xl-base-1.0": "~7 GB",
        "zai-org/CogView3-Plus-3B": "~6 GB",
        "PixArt-alpha/PixArt-XL-2-512x512": "~4 GB",
        "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "~4 GB",
        "Alpha-VLLM/Lumina-Next-SFT-diffusers": "~8 GB",
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": "~10 GB",
        "stabilityai/stable-diffusion-3-medium-diffusers": "~9 GB",
        "black-forest-labs/FLUX.1-dev": "~16 GB",
        "Efficient-Large-Model/Sana_600M_512px_diffusers": "~3 GB",
        "Qwen/Qwen-Image": "~8 GB",
        "thu-ml/unidiffuser-v1": "~5 GB",
        "stabilityai/stable-cascade": "~10 GB",
        "zai-org/CogView4-6B": "~11 GB",
    }
    return memory_estimates.get(model_id, "~Unknown")


def format_model_info(model_id: str, loaded: bool = False) -> str:
    """Format model information for display."""
    model_info = MODELS.get(model_id, {})
    name = model_info.get("short_name", model_id)
    memory = estimate_memory_usage(model_id)
    status = "✓ Loaded" if loaded else "Not loaded"

    return f"{name} ({memory}) - {status}"


def create_and_save_image_grid(
    image_paths: List[str], rows: Optional[int] = None, cols: Optional[int] = None, base_dir: str = "outputs"
) -> Optional[str]:
    """Create an image grid from a list of image paths."""
    if not image_paths:
        return None
    
    # Load images (skip failed ones)
    images = []
    for path in image_paths:
        try:
            images.append(Image.open(path))
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
    
    if not images:
        return None
    
    # Auto-calculate grid dimensions
    num_images = len(images)
    if rows is None and cols is None:
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
    elif rows is None:
        rows = math.ceil(num_images / cols)
    elif cols is None:
        cols = math.ceil(num_images / rows)
    
    try:
        grid_image = make_image_grid(images, rows=rows, cols=cols)
        output_dir = Path(base_dir) / "grids"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"image_grid_{rows}x{cols}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        grid_image.save(filepath)
        return str(filepath)
    except Exception as e:
        print(f"Failed to create image grid: {e}")
        return None
