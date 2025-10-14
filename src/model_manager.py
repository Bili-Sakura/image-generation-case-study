"""
Model manager for loading and managing diffusion pipelines.
"""

import torch
import os
from typing import Dict, List, Optional
import json
import importlib
from diffusers import DiffusionPipeline
from diffusers.pipelines.lumina.pipeline_lumina import LuminaPipeline
from diffusers.pipelines.stable_cascade.pipeline_stable_cascade import StableCascadeDecoderPipeline
from diffusers.pipelines.stable_cascade.pipeline_stable_cascade_prior import StableCascadePriorPipeline
import gc

from src.config import MODELS, DEFAULT_MODELS, LOCAL_MODEL_DIR, FLOW_MATCHING_MODELS, DIFFUSION_MODELS, SPECIAL_SCHEDULER_MODELS, USE_UNIFIED_SCHEDULER
from src.utils import get_device


class ModelManager:
    """Manages loading and caching of diffusion models."""

    def __init__(self, device: Optional[str] = None, use_device_map: bool = True):
        """Initialize model manager.

        Args:
            device: Device to load models on (cuda/cpu). Auto-detect if None.
            use_device_map: If True, use device_map="auto" for multi-GPU support.
        """
        self.device = device or get_device()
        self.use_device_map = use_device_map and torch.cuda.device_count() > 1
        self.loaded_models: Dict[str, DiffusionPipeline] = {}
        self.model_configs = MODELS
        self.local_model_dir = LOCAL_MODEL_DIR
        
        if self.use_device_map:
            print(f"Multi-GPU mode enabled: {torch.cuda.device_count()} GPUs detected")
        else:
            print(f"Single device mode: {self.device}")

    def _get_local_model_path(self, model_id: str) -> str:
        """Get local model path if exists, otherwise return original model_id."""
        if os.path.isabs(model_id):
            return model_id
        
        local_path = os.path.join(self.local_model_dir, model_id)
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
        return model_id

    def _get_pipeline_class(self, model_path: str) -> type:
        """Determine the pipeline class from the model's index file."""
        model_index_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(model_index_path):
            with open(model_index_path, "r") as f:
                config = json.load(f)
            class_name = config.get("_class_name")
            if class_name == "LuminaText2ImgPipeline":
                return LuminaPipeline
        return DiffusionPipeline

    def _apply_unified_scheduler(self, pipe: DiffusionPipeline, model_id: str):
        """Apply unified scheduler based on model type.
        
        Flow matching models -> FlowMatchEulerDiscreteScheduler
        Diffusion models -> EulerDiscreteScheduler
        Special models -> Keep original scheduler
        """
        if not USE_UNIFIED_SCHEDULER:
            return
        
        # Skip models with special schedulers
        if model_id in SPECIAL_SCHEDULER_MODELS:
            print(f"Keeping original scheduler for special model: {model_id}")
            return
        
        try:
            from diffusers import EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler
            
            if model_id in FLOW_MATCHING_MODELS:
                # Apply FlowMatchEulerDiscreteScheduler for flow matching models
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                    pipe.scheduler.config,
                    shift=3.0,  # Common shift value for flow matching
                )
                print(f"✓ Applied FlowMatchEulerDiscreteScheduler to {model_id}")
                
            elif model_id in DIFFUSION_MODELS:
                # Apply EulerDiscreteScheduler for diffusion models
                pipe.scheduler = EulerDiscreteScheduler.from_config(
                    pipe.scheduler.config,
                    timestep_spacing="trailing",
                )
                print(f"✓ Applied EulerDiscreteScheduler to {model_id}")
            else:
                # For models not explicitly categorized, try to auto-detect
                print(f"⚠️ Model {model_id} not categorized. Keeping original scheduler.")
                
        except Exception as e:
            print(f"⚠️ Could not apply unified scheduler to {model_id}: {e}")
            print(f"   Keeping original scheduler.")
    
    def _load_custom_scheduler(self, pipe: DiffusionPipeline, model_path: str):
        """Load a custom scheduler if defined in the model's directory."""
        scheduler_config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
        if os.path.exists(scheduler_config_path):
            with open(scheduler_config_path, "r") as f:
                scheduler_config = json.load(f)
            scheduler_class_name = scheduler_config.get("_class_name")
            if scheduler_class_name:
                try:
                    scheduler_module = importlib.import_module("diffusers.schedulers")
                    scheduler_class = getattr(scheduler_module, scheduler_class_name)
                    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                    print(f"Loaded custom scheduler: {scheduler_class_name}")
                except (ImportError, AttributeError) as e:
                    print(f"⚠️ Could not load custom scheduler {scheduler_class_name}: {e}")

    def load_model(self, model_id: str, force_reload: bool = False, use_device_map_override: Optional[bool] = None) -> DiffusionPipeline:
        """Load a specific model pipeline."""
        if model_id in self.loaded_models and not force_reload:
            print(f"Using cached model: {model_id}")
            return self.loaded_models[model_id]

        if model_id not in self.model_configs:
            raise ValueError(f"Unknown model ID: {model_id}")

        print(f"Loading model: {model_id}")
        
        # Determine whether to use device_map
        use_device_map_now = self.use_device_map if use_device_map_override is None else use_device_map_override
        
        try:
            local_model_path = self._get_local_model_path(model_id)

            pipeline_class = self._get_pipeline_class(local_model_path)
            
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                # "use_safetensors": True,
            }
            
            # Add device_map if multi-GPU mode is enabled
            if use_device_map_now:
                load_kwargs["device_map"] = "balanced"
            
            try:
                if model_id == "zai-org/CogView4-6B":
                    pipe = pipeline_class.from_pretrained(local_model_path, **load_kwargs).to(self.device)
                else:
                    pipe = pipeline_class.from_pretrained(local_model_path, **load_kwargs)
            except (AttributeError, TypeError, NotImplementedError) as e:
                # Some models don't work well with device_map, or accelerate is not installed
                # Fallback to regular loading
                if use_device_map_now and ("device_map" in str(e) or "_parameters" in str(e) or "accelerate" in str(e)):
                    print(f"⚠️  device_map failed for {model_id}, retrying without device_map...")
                    load_kwargs.pop("device_map", None)
                    pipe = pipeline_class.from_pretrained(local_model_path, **load_kwargs)
                    use_device_map_now = False
                else:
                    raise
            
            # Apply unified scheduler (EulerDiscreteScheduler or FlowMatchEulerDiscreteScheduler)
            self._apply_unified_scheduler(pipe, model_id)

            if not use_device_map_now and model_id != "zai-org/CogView4-6B":
                pipe = pipe.to(self.device)

            # Disable safety checker if not required
            if not self.model_configs[model_id].get("requires_safety_checker", False) and hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None
            
            if model_id == "zai-org/CogView4-6B":
                print("Applying CogView4-specific optimizations...")
                pipe.enable_model_cpu_offload()
                if hasattr(pipe, "vae"):
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()
            elif model_id in ["Alpha-VLLM/Lumina-Image-2.0", "HiDream-ai/HiDream-I1-Dev"]:
                print(f"Applying optimizations for {model_id}...")
                if not use_device_map_now:  # Only apply CPU offload if not using device_map
                    pipe.enable_model_cpu_offload()
                if hasattr(pipe, "vae"):
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()

            self.loaded_models[model_id] = pipe
            print(f"✓ Successfully loaded: {model_id}")
            return pipe
        except Exception as e:
            print(f"✗ Failed to load {model_id}: {str(e)}")
            raise

    def load_models(self, model_ids: List[str]) -> Dict[str, DiffusionPipeline]:
        """Load multiple models."""
        loaded = {}
        for model_id in model_ids:
            try:
                loaded[model_id] = self.load_model(model_id)
            except Exception as e:
                print(f"Skipping {model_id} due to error: {e}")
        return loaded

    def unload_model(self, model_id: str):
        """Unload a specific model from memory."""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Unloaded model: {model_id}")

    def unload_all_models(self):
        """Unload all models from memory."""
        self.loaded_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("All models unloaded")

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded."""
        return model_id in self.loaded_models

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs."""
        return list(self.loaded_models.keys())

    def get_pipeline(self, model_id: str) -> Optional[DiffusionPipeline]:
        """Get a loaded pipeline."""
        return self.loaded_models.get(model_id)

    def get_stable_cascade_prior_embeddings(
        self,
        pipe: StableCascadeDecoderPipeline,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ):
        """Load Stable Cascade prior pipeline and generate image embeddings.
        
        Args:
            pipe: The StableCascadeDecoderPipeline instance
            prompt: Text prompt
            negative_prompt: Negative prompt
            seed: Random seed
            
        Returns:
            Image embeddings tensor
        """
        prior_model_path = os.path.join(LOCAL_MODEL_DIR, "stabilityai/stable-cascade-prior")
        if not os.path.exists(prior_model_path):
            prior_model_path = "stabilityai/stable-cascade-prior"

        try:
            prior_pipeline = StableCascadePriorPipeline.from_pretrained(
                prior_model_path, torch_dtype=pipe.dtype, device_map="balanced"
            )
        except (NotImplementedError, Exception) as e:
            # Fallback without device_map if accelerate is not available
            if "accelerate" in str(e):
                print(f"⚠️  Loading Stable Cascade prior without device_map (accelerate not available)")
                prior_pipeline = StableCascadePriorPipeline.from_pretrained(
                    prior_model_path, torch_dtype=pipe.dtype
                ).to(self.device)
            else:
                raise
        prompt_embeds = prior_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_inference_steps=20,
            guidance_scale=4.0,
        ).image_embeddings
        
        return prompt_embeds


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def initialize_default_models() -> ModelManager:
    """Initialize model manager with default models.

    Returns:
        Initialized model manager
    """
    manager = get_model_manager()
    print("Initializing default models...")
    manager.load_models(DEFAULT_MODELS)
    return manager
