"""
Inference logic for text-to-image generation.
"""

import torch
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
import traceback
from diffusers import EulerDiscreteScheduler

from src.model_manager import get_model_manager
from src.config import MODELS, DEFAULT_CONFIG
from src.utils import set_seed, save_image


def generate_image(
    model_id: str,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: int = -1,
    scheduler: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Tuple[Optional[Image.Image], str, int]:
    """Generate a single image with a specific model."""
    manager = get_model_manager()
    model_config = MODELS.get(model_id, {})

    # Use defaults if not provided
    num_inference_steps = num_inference_steps or DEFAULT_CONFIG["num_inference_steps"]
    guidance_scale = guidance_scale or DEFAULT_CONFIG["guidance_scale"]
    width = width or DEFAULT_CONFIG["width"]
    height = height or DEFAULT_CONFIG["height"]
    seed_used = set_seed(seed)

    try:
        pipe = manager.get_pipeline(model_id) or manager.load_model(model_id)
        
        # Set scheduler if specified
        scheduler_name = scheduler or DEFAULT_CONFIG.get("scheduler")
        if scheduler_name == "EulerDiscreteScheduler" and hasattr(pipe, "scheduler"):
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        if progress_callback:
            progress_callback(f"Generating with {model_config.get('short_name', model_id)}...")

        # Prepare generation parameters
        generator_device = "cpu" if manager.use_device_map else manager.device
        gen_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": torch.Generator(device=generator_device).manual_seed(seed_used),
        }

        if negative_prompt and hasattr(pipe, "negative_prompt"):
            gen_kwargs["negative_prompt"] = negative_prompt

        image = pipe(**gen_kwargs).images[0]
        filepath = save_image(image, model_id, seed_used, prompt)

        if progress_callback:
            progress_callback(f"✓ Completed {model_config.get('short_name', model_id)}")

        return image, filepath, seed_used
    except Exception as e:
        error_msg = f"Error generating with {model_id}: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        if progress_callback:
            progress_callback(f"✗ Failed: {model_config.get('short_name', model_id)}")
        return None, error_msg, seed_used


def generate_images_sequential(
    model_ids: List[str],
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: int = -1,
    scheduler: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> List[Tuple[str, Optional[Image.Image], str, int]]:
    """Generate images sequentially with multiple models."""
    base_seed = set_seed(-1) if seed == -1 else seed
    results = []

    for i, model_id in enumerate(model_ids):
        if progress_callback:
            model_name = MODELS.get(model_id, {}).get('short_name', model_id)
            progress_callback(f"Processing: {model_name}", i + 1, len(model_ids))

        image, filepath, seed_used = generate_image(
            model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            width=width, height=height, seed=base_seed, scheduler=scheduler,
            progress_callback=None,  # Avoid nested callbacks
        )
        results.append((model_id, image, filepath, seed_used))

    return results


def batch_generate(model_ids: List[str], prompts: List[str], **kwargs) -> Dict[str, List[Tuple[Optional[Image.Image], str, int]]]:
    """Generate images for multiple prompts with multiple models."""
    results = {}
    for prompt in prompts:
        prompt_results = generate_images_sequential(model_ids=model_ids, prompt=prompt, **kwargs)
        results[prompt] = [(img, fp, s) for _, img, fp, s in prompt_results]
    return results
