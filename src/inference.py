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
    """Generate a single image with a specific model.

    Args:
        model_id: HuggingFace model ID
        prompt: Text prompt
        negative_prompt: Negative text prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        width: Image width
        height: Image height
        seed: Random seed (-1 for random)
        scheduler: Scheduler name (e.g., "EulerDiscreteScheduler")
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (generated_image, filepath, seed_used)
    """
    manager = get_model_manager()

    # Get model config for metadata
    model_config = MODELS.get(model_id, {})

    # Use DEFAULT_CONFIG if not provided
    if num_inference_steps is None:
        num_inference_steps = DEFAULT_CONFIG["num_inference_steps"]
    if guidance_scale is None:
        guidance_scale = DEFAULT_CONFIG["guidance_scale"]
    if width is None:
        width = DEFAULT_CONFIG["width"]
    if height is None:
        height = DEFAULT_CONFIG["height"]

    # Set seed
    seed_used = set_seed(seed)

    try:
        # Get or load pipeline
        pipe = manager.get_pipeline(model_id)
        if pipe is None:
            if progress_callback:
                progress_callback(f"Loading {model_id}...")
            pipe = manager.load_model(model_id)

        # Set scheduler (use provided scheduler or fall back to DEFAULT_CONFIG)
        scheduler_name = scheduler or DEFAULT_CONFIG.get("scheduler")
        if scheduler_name and hasattr(pipe, "scheduler"):
            if scheduler_name == "EulerDiscreteScheduler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(
                    pipe.scheduler.config
                )

        if progress_callback:
            progress_callback(
                f"Generating with {model_config.get('short_name', model_id)}..."
            )

        # Prepare generation parameters
        # For multi-GPU with device_map, use CPU generator; otherwise use device-specific generator
        generator_device = "cpu" if manager.use_device_map else manager.device
        
        gen_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": torch.Generator(device=generator_device).manual_seed(seed_used),
        }

        # Add negative prompt if supported and provided
        if negative_prompt and hasattr(pipe, "negative_prompt"):
            gen_kwargs["negative_prompt"] = negative_prompt

        # Generate image
        output = pipe(**gen_kwargs)
        image = output.images[0]

        # Save image
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
    """Generate images sequentially with multiple models.

    Args:
        model_ids: List of model IDs to use
        prompt: Text prompt
        negative_prompt: Negative text prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        width: Image width
        height: Image height
        seed: Random seed (-1 for random)
        scheduler: Scheduler name (e.g., "EulerDiscreteScheduler")
        progress_callback: Optional callback for progress updates

    Returns:
        List of tuples (model_id, image, filepath, seed)
    """
    results = []

    # Use the same seed for all models if specified
    base_seed = seed
    if base_seed == -1:
        base_seed = set_seed(-1)

    for i, model_id in enumerate(model_ids):
        if progress_callback:
            # Get model short name for display
            model_name = MODELS.get(model_id, {}).get('short_name', model_id)
            progress_callback(f"Processing: {model_name}", i + 1, len(model_ids))

        image, filepath, seed_used = generate_image(
            model_id=model_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=base_seed,
            scheduler=scheduler,
            progress_callback=None,  # Don't pass nested callbacks to avoid conflicts
        )

        results.append((model_id, image, filepath, seed_used))

    return results


def batch_generate(
    model_ids: List[str], prompts: List[str], **kwargs
) -> Dict[str, List[Tuple[Optional[Image.Image], str, int]]]:
    """Generate images for multiple prompts with multiple models.

    Args:
        model_ids: List of model IDs
        prompts: List of prompts
        **kwargs: Additional generation parameters

    Returns:
        Dictionary mapping prompt to list of (image, filepath, seed) tuples
    """
    results = {}

    for prompt in prompts:
        prompt_results = generate_images_sequential(
            model_ids=model_ids, prompt=prompt, **kwargs
        )
        results[prompt] = [(img, fp, s) for _, img, fp, s in prompt_results]

    return results
