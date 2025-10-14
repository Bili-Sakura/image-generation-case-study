"""
Inference logic for text-to-image generation.
"""

import torch
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
import traceback
import os
from diffusers import EulerDiscreteScheduler
from diffusers.pipelines.stable_cascade.pipeline_stable_cascade import StableCascadeDecoderPipeline
from diffusers.pipelines.stable_cascade.pipeline_stable_cascade_prior import StableCascadePriorPipeline
import importlib


from src.model_manager import get_model_manager
from src.config import MODELS, DEFAULT_CONFIG, LOCAL_MODEL_DIR
from src.utils import save_image, seed_everything, get_timestamp_output_dir, save_generation_config
from pathlib import Path


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
    output_dir: Optional[Path] = None,
    model_manager = None,
) -> Tuple[Optional[Image.Image], str, int]:
    """Generate a single image with a specific model."""
    manager = model_manager if model_manager is not None else get_model_manager()
    model_config = MODELS.get(model_id, {})

    # Use defaults if not provided
    num_inference_steps = num_inference_steps or DEFAULT_CONFIG["num_inference_steps"]
    guidance_scale = guidance_scale or DEFAULT_CONFIG["guidance_scale"]
    width = width or DEFAULT_CONFIG["width"]
    height = height or DEFAULT_CONFIG["height"]
    seed_used = seed_everything(seed)

    # Create output directory if not provided
    if output_dir is None:
        output_dir = get_timestamp_output_dir()

    try:
        pipe = manager.get_pipeline(model_id) or manager.load_model(model_id)
        
        # Set scheduler if specified
        if scheduler and hasattr(pipe, "scheduler"):
            try:
                scheduler_class = getattr(importlib.import_module("diffusers.schedulers"), scheduler)
                pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                print(f"Using custom scheduler: {scheduler}")
            except (ImportError, AttributeError):
                print(f"Warning: Could not find or load scheduler '{scheduler}'. Using model's default.")

        if progress_callback:
            progress_callback(f"Generating with {model_config.get('short_name', model_id)}...", 0, 1)

        # Prepare generation parameters
        generator_device = "cpu" if manager.use_device_map else manager.device
        gen_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "generator": torch.Generator(device=generator_device).manual_seed(seed_used),
        }
        
        # Use true_cfg_scale for Qwen-Image, guidance_scale for other models
        if model_id == "Qwen/Qwen-Image":
            gen_kwargs["true_cfg_scale"] = guidance_scale
        elif model_id == "Alpha-VLLM/Lumina-Image-2.0":
            # Lumina-Image-2.0 supports additional CFG parameters
            gen_kwargs["guidance_scale"] = guidance_scale
            gen_kwargs["cfg_trunc_ratio"] = 0.25
            gen_kwargs["cfg_normalization"] = True
        else:
            gen_kwargs["guidance_scale"] = guidance_scale

        if negative_prompt and hasattr(pipe, "negative_prompt"):
            gen_kwargs["negative_prompt"] = negative_prompt

        if isinstance(pipe, StableCascadeDecoderPipeline):
            prior_model_path = os.path.join(LOCAL_MODEL_DIR, "stabilityai/stable-cascade-prior")
            if not os.path.exists(prior_model_path):
                prior_model_path = "stabilityai/stable-cascade-prior"

            prior_pipeline = StableCascadePriorPipeline.from_pretrained(
                prior_model_path, torch_dtype=pipe.dtype, device_map="balanced"
            )
            prompt_embeds = prior_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=torch.Generator(device="cpu").manual_seed(seed_used),
                num_inference_steps=20,
                guidance_scale=4.0,
            ).image_embeddings
            gen_kwargs["image_embeddings"] = prompt_embeds
            gen_kwargs.pop("width", None)
            gen_kwargs.pop("height", None)
            gen_kwargs.pop("guidance_scale", None)

        image = pipe(**gen_kwargs).images[0]
        filepath = save_image(image, model_id, seed_used, output_dir)

        if progress_callback:
            progress_callback(f"✓ Completed {model_config.get('short_name', model_id)}", 0, 1)

        return image, filepath, seed_used
    except Exception as e:
        error_msg = f"Error generating with {model_id}: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        if progress_callback:
            progress_callback(f"✗ Failed: {model_config.get('short_name', model_id)}", 0, 1)
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
    model_manager = None,
) -> List[Tuple[str, Optional[Image.Image], str, int]]:
    """Generate images sequentially with multiple models."""
    from datetime import datetime
    
    base_seed = seed_everything(-1) if seed == -1 else seed
    
    # Create single timestamp directory for all outputs
    output_dir = get_timestamp_output_dir()
    
    # Use defaults if not provided
    num_inference_steps = num_inference_steps or DEFAULT_CONFIG["num_inference_steps"]
    guidance_scale = guidance_scale or DEFAULT_CONFIG["guidance_scale"]
    width = width or DEFAULT_CONFIG["width"]
    height = height or DEFAULT_CONFIG["height"]
    
    results = []
    model_results = []

    for i, model_id in enumerate(model_ids):
        if progress_callback:
            model_name = MODELS.get(model_id, {}).get('short_name', model_id)
            progress_callback(f"Processing: {model_name}", i + 1, len(model_ids))

        image, filepath, seed_used = generate_image(
            model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            width=width, height=height, seed=base_seed, scheduler=scheduler,
            progress_callback=None,  # Avoid nested callbacks
            output_dir=output_dir,
            model_manager=model_manager,
        )
        results.append((model_id, image, filepath, seed_used))
        
        # Collect model info for config
        if image is not None:
            model_results.append({
                "model_id": model_id,
                "model_name": MODELS.get(model_id, {}).get("short_name", model_id),
                "image_path": filepath,
                "seed_used": seed_used,
            })

    # Save generation config JSON
    config_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "parameters": {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": base_seed,
            "scheduler": scheduler,
        },
        "models": model_results,
        "output_directory": str(output_dir),
    }
    
    save_generation_config(output_dir, config_data)

    return results


def batch_generate(model_ids: List[str], prompts: List[str], **kwargs) -> Dict[str, List[Tuple[Optional[Image.Image], str, int]]]:
    """Generate images for multiple prompts with multiple models."""
    results = {}
    for prompt in prompts:
        prompt_results = generate_images_sequential(model_ids=model_ids, prompt=prompt, **kwargs)
        results[prompt] = [(img, fp, s) for _, img, fp, s in prompt_results]
    return results


def generate_all_models_sequential(
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
    """Generate images with all available models (load → inference → unload for each).
    
    This function processes all models sequentially without pre-loading,
    loading and unloading each model one by one to minimize memory usage.
    
    Args:
        prompt: Text prompt
        negative_prompt: Negative prompt
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        width: Image width
        height: Image height
        seed: Random seed (-1 for random)
        scheduler: Optional scheduler name
        progress_callback: Optional progress callback function
        
    Returns:
        List of (model_id, image, filepath, seed_used) tuples
    """
    from datetime import datetime
    
    # Get all available model IDs
    model_ids = list(MODELS.keys())
    
    base_seed = seed_everything(-1) if seed == -1 else seed
    
    # Create single timestamp directory for all outputs
    output_dir = get_timestamp_output_dir()
    
    # Use defaults if not provided
    num_inference_steps = num_inference_steps or DEFAULT_CONFIG["num_inference_steps"]
    guidance_scale = guidance_scale or DEFAULT_CONFIG["guidance_scale"]
    width = width or DEFAULT_CONFIG["width"]
    height = height or DEFAULT_CONFIG["height"]
    
    results = []
    model_results = []
    
    # Create a dedicated model manager for batch processing
    model_manager = get_model_manager()

    for i, model_id in enumerate(model_ids):
        if progress_callback:
            model_name = MODELS.get(model_id, {}).get('short_name', model_id)
            progress_callback(f"[{i+1}/{len(model_ids)}] Loading: {model_name}", i + 1, len(model_ids))

        try:
            # Load the model
            model_manager.load_model(model_id, force_reload=False)
            
            if progress_callback:
                model_name = MODELS.get(model_id, {}).get('short_name', model_id)
                progress_callback(f"[{i+1}/{len(model_ids)}] Generating: {model_name}", i + 1, len(model_ids))
            
            # Generate image
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
                progress_callback=None,  # Avoid nested callbacks
                output_dir=output_dir,
                model_manager=model_manager,
            )
            
            results.append((model_id, image, filepath, seed_used))
            
            # Collect model info for config
            if image is not None:
                model_results.append({
                    "model_id": model_id,
                    "model_name": MODELS.get(model_id, {}).get("short_name", model_id),
                    "image_path": filepath,
                    "seed_used": seed_used,
                })
            
            if progress_callback:
                model_name = MODELS.get(model_id, {}).get('short_name', model_id)
                progress_callback(f"[{i+1}/{len(model_ids)}] Unloading: {model_name}", i + 1, len(model_ids))
            
            # Unload the model to free memory
            model_manager.unload_model(model_id)
            
        except Exception as e:
            error_msg = f"Error with {model_id}: {str(e)}"
            print(error_msg)
            results.append((model_id, None, error_msg, base_seed))
            # Try to unload even if there was an error
            try:
                model_manager.unload_model(model_id)
            except:
                pass

    # Save generation config JSON
    config_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "parameters": {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": base_seed,
            "scheduler": scheduler,
        },
        "models": model_results,
        "output_directory": str(output_dir),
        "mode": "benchmark_all_models",
    }
    
    save_generation_config(output_dir, config_data)

    return results
