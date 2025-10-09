"""
Utility functions for image processing and file handling.
"""

from datetime import datetime
from pathlib import Path
import torch
from PIL import Image


def setup_output_dir(base_dir: str = "outputs") -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(base_dir)
    output_path.mkdir(exist_ok=True)
    return output_path


def get_model_output_dir(model_id: str, base_dir: str = "outputs") -> Path:
    """Get or create output directory for a specific model."""
    # Sanitize model_id for directory name
    dir_name = model_id.replace("/", "-")
    output_path = Path(base_dir) / dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_image(
    image: Image.Image, model_id: str, seed: int, prompt: str, base_dir: str = "outputs"
) -> str:
    """Save generated image with metadata."""
    output_dir = get_model_output_dir(model_id, base_dir)

    # Create filename with timestamp and seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_seed{seed}.png"
    filepath = output_dir / filename

    # Save image
    image.save(filepath)

    # Save metadata
    metadata_file = filepath.with_suffix(".txt")
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Timestamp: {timestamp}\n")

    return str(filepath)


def set_seed(seed: int) -> int:
    """Set random seed for reproducibility. Returns the seed used."""
    if seed == -1:
        import random

        seed = random.randint(0, 2**32 - 1)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def get_device() -> str:
    """Get the appropriate device (cuda/cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


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
    from src.config import MODELS

    model_info = MODELS.get(model_id, {})
    name = model_info.get("short_name", model_id)
    memory = estimate_memory_usage(model_id)
    status = "✓ Loaded" if loaded else "Not loaded"

    return f"{name} ({memory}) - {status}"
