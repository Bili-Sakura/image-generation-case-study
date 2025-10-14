"""
Utility functions for image processing and file handling.
"""

from datetime import datetime
from pathlib import Path
import torch
import math
import json
from PIL import Image
from typing import List, Optional, Dict, Any
from diffusers.utils import make_image_grid
from src.config import MODELS


def get_timestamp_output_dir(base_dir: str = "outputs") -> Path:
    """Get or create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(base_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_image(
    image: Image.Image, 
    model_id: str, 
    seed: int, 
    output_dir: Path
) -> str:
    """Save generated image to specified directory.
    
    Args:
        image: PIL Image to save
        model_id: Model identifier
        seed: Random seed used
        output_dir: Directory to save the image
    
    Returns:
        Path to saved image
    """
    model_short_name = MODELS.get(model_id, {}).get("short_name", model_id)
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_short_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filepath = output_dir / f"{safe_name}_{timestamp}_seed{seed}.png"
    image.save(filepath)
    
    return str(filepath)


def save_generation_config(
    output_dir: Path,
    config_data: Dict[str, Any]
) -> str:
    """Save generation configuration to JSON file.
    
    Args:
        output_dir: Directory to save the config
        config_data: Configuration data including prompt, models, params, etc.
    
    Returns:
        Path to saved JSON file
    """
    filepath = output_dir / "generation_config.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    return str(filepath)


def seed_everything(seed: int) -> int:
    """Set random seed for reproducibility across all libraries."""
    import random
    import numpy as np
    
    seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    
    random.seed(seed)
    np.random.seed(seed)
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


def get_gpu_vram_usage() -> str:
    """Get current GPU VRAM usage for all GPUs."""
    if not torch.cuda.is_available():
        return "⚠️ No CUDA GPUs available"
    
    gpu_count = torch.cuda.device_count()
    info_lines = []
    
    for i in range(gpu_count):
        torch.cuda.reset_peak_memory_stats(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        free = total - reserved
        usage_percent = (reserved / total) * 100
        
        # Create progress bar
        bar_length = 20
        filled = int(bar_length * usage_percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        info_lines.append(
            f"GPU {i}: [{bar}] {usage_percent:.1f}%\n"
            f"  Used: {reserved:.2f} GB / {total:.1f} GB | Free: {free:.2f} GB\n"
            f"  Allocated: {allocated:.2f} GB"
        )
    
    return "\n\n".join(info_lines)


# Model parameter and VRAM data
MODEL_PARAMS_DATA = {
    "stabilityai/stable-diffusion-2-1": {"params": 1.29, "components": {"text_encoder": 0.34039, "unet": 0.86591, "vae": 0.08365}},
    "stabilityai/stable-diffusion-2-1-base": {"params": 1.29, "components": {"text_encoder": 0.34039, "unet": 0.86591, "vae": 0.08365}},
    "stabilityai/stable-diffusion-xl-base-1.0": {"params": 3.47, "components": {"text_encoder": 0.12306, "text_encoder_2": 0.69466, "unet": 2.57, "vae": 0.08365}},
    "zai-org/CogView3-Plus-3B": {"params": 8.02, "components": {"text_encoder": 4.76, "transformer": 2.85, "vae": 0.40610}},
    "PixArt-alpha/PixArt-XL-2-512x512": {"params": 5.46, "components": {"text_encoder": 4.76, "transformer": 0.61086, "vae": 0.08365}},
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": {"params": 5.46, "components": {"text_encoder": 4.76, "transformer": 0.61086, "vae": 0.08365}},
    "Alpha-VLLM/Lumina-Next-SFT-diffusers": {"params": 4.34, "components": {"text_encoder": 2.51, "transformer": 1.75, "vae": 0.08365}},
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": {"params": 3.61, "components": {"text_encoder": 0.35204, "text_encoder_2": 1.67, "transformer": 1.50, "vae": 0.08365}},
    "stabilityai/stable-diffusion-3-medium-diffusers": {"params": 7.69, "components": {"text_encoder": 0.12365, "text_encoder_2": 0.69466, "text_encoder_3": 4.76, "transformer": 2.03, "vae": 0.08382}},
    "black-forest-labs/FLUX.1-dev": {"params": 16.87, "components": {"text_encoder": 0.12306, "text_encoder_2": 4.76, "transformer": 11.90, "vae": 0.08382}},
    "Efficient-Large-Model/Sana_600M_512px_diffusers": {"params": 3.52, "components": {"text_encoder": 2.61, "transformer": 0.59175, "vae": 0.31225}},
    "Qwen/Qwen-Image": {"params": 28.85, "components": {"text_encoder": 8.29, "transformer": 20.43, "vae": 0.12689}},
    "thu-ml/unidiffuser-v1": {"params": 1.25, "components": {"image_encoder": 0.08785, "text_encoder": 0.12306, "unet": 0.95254, "vae": 0.08365}},
    "stabilityai/stable-cascade": {"params": 2.28, "components": {"decoder": 1.56, "text_encoder": 0.69466, "vqgan": 0.01841}},
    "zai-org/CogView4-6B": {"params": 6.0, "components": {"text_encoder": 2.0, "transformer": 3.5, "vae": 0.5}},
}


def estimate_memory_usage(model_id: str) -> str:
    """Estimate VRAM usage for a model based on parameter count.
    
    VRAM = 2 × parameters (in billions) for bfloat16 models.
    """
    params_data = MODEL_PARAMS_DATA.get(model_id)
    if params_data:
        vram_gb = params_data["params"] * 2
        return f"{vram_gb:.1f} GB"
    return "~Unknown"


def get_model_params_table() -> str:
    """Generate a markdown table of model parameters and VRAM usage."""
    from src.config import MODELS
    
    table_lines = [
        "| Model | Parameters (B) | VRAM (bfloat16) | Components |",
        "|-------|----------------|-----------------|------------|"
    ]
    
    # Sort by parameters (smallest to largest)
    sorted_models = sorted(
        MODEL_PARAMS_DATA.items(),
        key=lambda x: x[1]["params"]
    )
    
    for model_id, data in sorted_models:
        # Get short name from MODELS config
        model_info = MODELS.get(model_id, {})
        short_name = model_info.get("short_name", model_id.split("/")[-1])
        
        params = data["params"]
        vram = params * 2
        
        # Format components
        components = data["components"]
        comp_str = ", ".join([f"{k}: {v:.2f}B" for k, v in components.items()])
        
        table_lines.append(
            f"| {short_name} | {params:.2f} | {vram:.1f} GB | {comp_str} |"
        )
    
    # Add footer note
    table_lines.append("")
    table_lines.append("**Note:** VRAM estimate = 2 × Parameters (in billions) for bfloat16 precision.")
    table_lines.append("Actual usage may vary depending on batch size, resolution, and optimization techniques.")
    
    return "\n".join(table_lines)


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
