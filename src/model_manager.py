"""
Model manager for loading and managing diffusion pipelines.
"""

import torch
import os
from typing import Dict, List, Optional
from diffusers import DiffusionPipeline
import gc

from src.config import MODELS, DEFAULT_MODELS, LOCAL_MODEL_DIR
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
        """Get the local model path if it exists, otherwise return the original model_id.

        Args:
            model_id: HuggingFace model ID or local path

        Returns:
            Local path if it exists, otherwise the original model_id
        """
        # If it's already a full path, return as is
        if os.path.isabs(model_id):
            return model_id

        # Check if model exists in local directory
        local_path = os.path.join(self.local_model_dir, model_id)

        # If model exists locally, use the local path
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path

        # Otherwise, use the original HuggingFace model ID
        return model_id

    def load_model(
        self, model_id: str, force_reload: bool = False
    ) -> DiffusionPipeline:
        """Load a specific model pipeline.

        Args:
            model_id: HuggingFace model ID
            force_reload: Force reload even if already cached

        Returns:
            Loaded diffusion pipeline
        """
        # Return cached model if available
        if model_id in self.loaded_models and not force_reload:
            print(f"Using cached model: {model_id}")
            return self.loaded_models[model_id]

        # Validate model ID
        if model_id not in self.model_configs:
            raise ValueError(f"Unknown model ID: {model_id}")

        print(f"Loading model: {model_id}")

        try:
            # Check if local model exists
            local_model_path = self._get_local_model_path(model_id)

            # Prepare loading arguments
            load_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "use_safetensors": True,
            }
            
            # Use device_map="auto" for multi-GPU, otherwise specify device
            if self.use_device_map:
                load_kwargs["device_map"] = "auto"
                print(f"Loading {model_id} with device_map='auto' for multi-GPU support")
            
            # Load the pipeline from local path if available, otherwise from HuggingFace
            pipe = DiffusionPipeline.from_pretrained(
                local_model_path,
                **load_kwargs
            )

            # Move to specific device only if not using device_map
            if not self.use_device_map:
                pipe = pipe.to(self.device)

            # Disable safety checker if not required (for faster inference)
            model_config = self.model_configs[model_id]
            if not model_config.get("requires_safety_checker", False):
                if hasattr(pipe, "safety_checker"):
                    pipe.safety_checker = None

            # Cache the model
            self.loaded_models[model_id] = pipe

            print(f"✓ Successfully loaded: {model_id}")
            return pipe

        except Exception as e:
            print(f"✗ Failed to load {model_id}: {str(e)}")
            raise

    def load_models(self, model_ids: List[str]) -> Dict[str, DiffusionPipeline]:
        """Load multiple models.

        Args:
            model_ids: List of model IDs to load

        Returns:
            Dictionary of loaded pipelines
        """
        loaded = {}
        for model_id in model_ids:
            try:
                pipe = self.load_model(model_id)
                loaded[model_id] = pipe
            except Exception as e:
                print(f"Skipping {model_id} due to error: {e}")
                continue

        return loaded

    def unload_model(self, model_id: str):
        """Unload a specific model from memory.

        Args:
            model_id: Model ID to unload
        """
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
        """Check if a model is loaded.

        Args:
            model_id: Model ID to check

        Returns:
            True if model is loaded
        """
        return model_id in self.loaded_models

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs.

        Returns:
            List of loaded model IDs
        """
        return list(self.loaded_models.keys())

    def get_pipeline(self, model_id: str) -> Optional[DiffusionPipeline]:
        """Get a loaded pipeline.

        Args:
            model_id: Model ID

        Returns:
            Pipeline if loaded, None otherwise
        """
        return self.loaded_models.get(model_id)


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
