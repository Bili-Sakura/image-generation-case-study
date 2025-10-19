"""
Inference logic for text-to-image generation.
"""

import torch
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
import traceback
import os
import time
from diffusers import EulerDiscreteScheduler, KandinskyPipeline
from diffusers.pipelines.stable_cascade.pipeline_stable_cascade import StableCascadeDecoderPipeline
import importlib


from src.model_manager import get_model_manager
from src.config import MODELS, DEFAULT_CONFIG, ENABLE_COMPUTE_PROFILING
from src.utils import save_image, seed_everything, get_timestamp_output_dir, save_generation_config
from src.compute_profiler import create_profiler
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
    enable_profiling: Optional[bool] = None,
    pipe: Optional[Callable] = None,
) -> Tuple[Optional[Image.Image], str, int, Optional[Dict]]:
    """Generate a single image with a specific model.
    
    Returns:
        Tuple of (image, filepath, seed_used, profiling_data)
        profiling_data is None if profiling is disabled
    """
    manager = model_manager if model_manager is not None else get_model_manager()
    model_config = MODELS.get(model_id, {})

    # Use defaults if not provided
    num_inference_steps = num_inference_steps or DEFAULT_CONFIG["num_inference_steps"]
    guidance_scale = guidance_scale or DEFAULT_CONFIG["guidance_scale"]
    width = width or DEFAULT_CONFIG["width"]
    height = height or DEFAULT_CONFIG["height"]
    seed_used = seed_everything(seed)
    
    # Enable profiling by default from config
    if enable_profiling is None:
        enable_profiling = DEFAULT_CONFIG.get("enable_profiling", ENABLE_COMPUTE_PROFILING)

    # Create output directory if not provided
    if output_dir is None:
        output_dir = get_timestamp_output_dir()
    
    # Initialize profiler
    profiler = create_profiler(enabled=True) if enable_profiling else None
    profiling_data = None

    try:
        if pipe is None:
            pipe = manager.get_pipeline(model_id) or manager.load_model(model_id)
        
        # Override scheduler if explicitly specified (overrides unified scheduler)
        if scheduler and hasattr(pipe, "scheduler"):
            try:
                scheduler_class = getattr(importlib.import_module("diffusers.schedulers"), scheduler)
                pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                print(f"Using custom scheduler override: {scheduler}")
            except (ImportError, AttributeError):
                print(f"Warning: Could not find or load scheduler '{scheduler}'. Using unified scheduler.")

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
            prompt_embeds = manager.get_stable_cascade_prior_embeddings(
                pipe, prompt, negative_prompt, seed_used
            )
            gen_kwargs["image_embeddings"] = prompt_embeds
            gen_kwargs.pop("width", None)
            gen_kwargs.pop("height", None)
            gen_kwargs.pop("guidance_scale", None)
        
        if isinstance(pipe, KandinskyPipeline):
            image_embeds, negative_image_embeds = manager.get_kandinsky_prior_embeddings(
                pipe, prompt, negative_prompt, seed_used, guidance_scale
            )
            gen_kwargs["image_embeds"] = image_embeds
            gen_kwargs["negative_image_embeds"] = negative_image_embeds
        
        # Profile compute cost before generation
        if enable_profiling:
            input_shape = (1, 3, height, width)  # Batch size 1, RGB image
            profiling_data = profiler.profile_pipeline(
                pipe=pipe,
                input_shape=input_shape,
                num_inference_steps=num_inference_steps,
                model_id=model_id,
            )
            
            # Print profiling summary
            if profiling_data.get("enabled"):
                print(f"\n📊 Compute Profile for {model_config.get('short_name', model_id)}:")
                print(f"   Parameters: {profiling_data.get('params_str', 'N/A')}")
                print(f"   FLOPs (total): {profiling_data.get('total_flops_str', 'N/A')}")
                print(f"   MACs (total): {profiling_data.get('total_macs_str', 'N/A')}")
                print(f"   FLOPs/step: {profiling_data.get('flops_per_step_str', 'N/A')}")
                print(f"   MACs/step: {profiling_data.get('macs_per_step_str', 'N/A')}")
                print(f"   Steps: {profiling_data.get('num_inference_steps', 'N/A')}\n")

        # Measure actual inference time
        start_time = time.time()
        image = pipe(**gen_kwargs).images[0]
        inference_time = time.time() - start_time
        
        # Add inference time to profiling data
        if profiling_data:
            profiling_data["inference_time_seconds"] = inference_time
            profiling_data["inference_time_str"] = f"{inference_time:.2f}s"
            print(f"⏱️  Inference time: {profiling_data['inference_time_str']}")
        
        filepath = save_image(image, model_id, seed_used, output_dir)

        if progress_callback:
            progress_callback(f"✓ Completed {model_config.get('short_name', model_id)}", 0, 1)

        return image, filepath, seed_used, profiling_data
    except Exception as e:
        error_msg = f"Error generating with {model_id}: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        if progress_callback:
            progress_callback(f"✗ Failed: {model_config.get('short_name', model_id)}", 0, 1)
        return None, error_msg, seed_used, None


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
    pipe: Optional[Callable] = None,
    unload_after_inference: bool = False,
) -> List[Tuple[str, Optional[Image.Image], str, int]]:
    """Generate images sequentially with multiple models.
    
    Args:
        unload_after_inference: If True, unload each model after inference to free memory.
    """
    from datetime import datetime
    
    base_seed = seed_everything(-1) if seed == -1 else seed
    
    # Create single timestamp directory for all outputs
    output_dir = get_timestamp_output_dir()
    
    # Use defaults if not provided
    num_inference_steps = num_inference_steps or DEFAULT_CONFIG["num_inference_steps"]
    guidance_scale = guidance_scale or DEFAULT_CONFIG["guidance_scale"]
    width = width or DEFAULT_CONFIG["width"]
    height = height or DEFAULT_CONFIG["height"]
    
    # Get model manager instance for unloading
    manager = model_manager if model_manager is not None else get_model_manager()
    
    results = []
    model_results = []

    for i, model_id in enumerate(model_ids):
        if progress_callback:
            model_name = MODELS.get(model_id, {}).get('short_name', model_id)
            progress_callback(f"Processing: {model_name}", i + 1, len(model_ids))

        image, filepath, seed_used, profiling_data = generate_image(
            model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            width=width, height=height, seed=base_seed, scheduler=scheduler,
            progress_callback=None,  # Avoid nested callbacks
            output_dir=output_dir,
            model_manager=manager,
            pipe=pipe,
        )
        results.append((model_id, image, filepath, seed_used))
        
        # Unload model after inference if requested (for benchmark mode)
        if unload_after_inference and image is not None:
            if progress_callback:
                model_name = MODELS.get(model_id, {}).get('short_name', model_id)
                progress_callback(f"Unloading: {model_name}", i + 1, len(model_ids))
            manager.unload_model(model_id)
            print(f"🗑️  Freed memory for: {MODELS.get(model_id, {}).get('short_name', model_id)}")
        
        # Collect model info for config
        if image is not None:
            model_info = {
                "model_id": model_id,
                "model_name": MODELS.get(model_id, {}).get("short_name", model_id),
                "image_path": filepath,
                "seed_used": seed_used,
            }
            
            # Add profiling data if available
            if profiling_data and profiling_data.get("enabled"):
                model_info["compute_profile"] = {
                    "total_params": profiling_data.get("total_params"),
                    "params_str": profiling_data.get("params_str"),
                    "total_flops": profiling_data.get("total_flops"),
                    "total_flops_str": profiling_data.get("total_flops_str"),
                    "total_macs": profiling_data.get("total_macs"),
                    "total_macs_str": profiling_data.get("total_macs_str"),
                    "flops_per_step": profiling_data.get("flops_per_step"),
                    "flops_per_step_str": profiling_data.get("flops_per_step_str"),
                    "macs_per_step": profiling_data.get("macs_per_step"),
                    "macs_per_step_str": profiling_data.get("macs_per_step_str"),
                    "inference_time_seconds": profiling_data.get("inference_time_seconds"),
                    "inference_time_str": profiling_data.get("inference_time_str"),
                    "model_component": profiling_data.get("model_component"),
                }
            
            model_results.append(model_info)

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
    model_ids = list(MODELS.keys())
    
    return generate_images_sequential(
        model_ids=model_ids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        scheduler=scheduler,
        progress_callback=progress_callback,
        unload_after_inference=True,  # Enable unload for benchmark mode
    )
