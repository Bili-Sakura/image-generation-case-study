"""
Configuration file for model definitions and default parameters.
"""

# Model definitions with metadata
MODELS = {
    "stabilityai/stable-diffusion-2-1-base": {
        "name": "Stable Diffusion 2.1",
        "short_name": "SD 2.1",
        "requires_safety_checker": False,
        "pipeline_class": "StableDiffusionPipeline",
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "name": "Stable Diffusion XL",
        "short_name": "SDXL",
        "requires_safety_checker": False,
        "pipeline_class": "StableDiffusionXLPipeline",
    },
    "zai-org/CogView3-Plus-3B": {
        "name": "CogView3 Plus 3B",
        "short_name": "CogView3+",
        "requires_safety_checker": False,
        "pipeline_class": "CogView3PlusPipeline",
    },
    "PixArt-alpha/PixArt-XL-2-512x512": {
        "name": "PixArt-XL 2",
        "short_name": "PixArt-Alpha",
        "requires_safety_checker": False,
        "pipeline_class": "PixArtAlphaPipeline",
    },
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": {
        "name": "PixArt-Sigma XL 2",
        "short_name": "PixArt-Sigma",
        "requires_safety_checker": False,
        "pipeline_class": "PixArtSigmaPipeline",
    },
    "Alpha-VLLM/Lumina-Next-SFT-diffusers": {
        "name": "Lumina-Next SFT",
        "short_name": "Lumina-Next",
        "requires_safety_checker": False,
        "pipeline_class": "LuminaText2ImgPipeline",
    },
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": {
        "name": "HunyuanDiT v1.2",
        "short_name": "HunyuanDiT",
        "requires_safety_checker": False,
        "pipeline_class": "HunyuanDiT2DPipeline",
    },
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "name": "Stable Diffusion 3 Medium",
        "short_name": "SD 3",
        "requires_safety_checker": False,
        "pipeline_class": "StableDiffusion3Pipeline",
    },
    "black-forest-labs/FLUX.1-dev": {
        "name": "FLUX.1 Dev",
        "short_name": "FLUX.1-dev",
        "requires_safety_checker": False,
        "pipeline_class": "FluxPipeline",
    },
    "Efficient-Large-Model/Sana_600M_512px_diffusers": {
        "name": "Sana 600M 512px",
        "short_name": "SANA",
        "requires_safety_checker": False,
        "pipeline_class": "SanaPipeline",
    },
    "Qwen/Qwen-Image": {
        "name": "Qwen Image",
        "short_name": "Qwen-Image",
        "requires_safety_checker": False,
        "pipeline_class": "DiffusionPipeline",
    },
    "thu-ml/unidiffuser-v1": {
        "name": "UniDiffuser v1",
        "short_name": "UniDiffuser",
        "requires_safety_checker": False,
        "pipeline_class": "UniDiffuserPipeline",
    },
    "stabilityai/stable-cascade": {
        "name": "Stable Cascade",
        "short_name": "Stable Cascade",
        "requires_safety_checker": False,
        "pipeline_class": "StableCascadeCombinedPipeline",
    },
    "zai-org/CogView4-6B": {
        "name": "CogView4 6B",
        "short_name": "CogView4",
        "requires_safety_checker": False,
        "pipeline_class": "DiffusionPipeline",
    },
    "kandinsky-community/kandinsky-3": {
        "name": "Kandinsky 3",
        "short_name": "Kandinsky-3",
        "requires_safety_checker": False,
        "pipeline_class": "Kandinsky3Pipeline",
    },
    "HiDream-ai/HiDream-I1-Dev": {
        "name": "HiDream I1 Dev",
        "short_name": "HiDream-I1",
        "requires_safety_checker": False,
        "pipeline_class": "HiDreamImagePipeline",
    },
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers": {
        "name": "SANA 1.5 1.6B 1024px",
        "short_name": "SANA 1.5",
        "requires_safety_checker": False,
        "pipeline_class": "SanaPipeline",
    },
    "Alpha-VLLM/Lumina-Image-2.0": {
        "name": "Lumina Image 2.0",
        "short_name": "Lumina-Image-2.0",
        "requires_safety_checker": False,
        "pipeline_class": "Lumina2Pipeline",
    },
}

# Default models configuration (deprecated - no longer used since user mode was removed)
# Models are now loaded manually in developer mode or automatically in batch mode
DEFAULT_MODELS = []

# Unified scheduler configuration
# Models are categorized by their training paradigm
FLOW_MATCHING_MODELS = [
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "Alpha-VLLM/Lumina-Next-SFT-diffusers",
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    "Qwen/Qwen-Image",
    "Alpha-VLLM/Lumina-Image-2.0",
    "zai-org/CogView4-6B",
    "HiDream-ai/HiDream-I1-Dev",
]

DIFFUSION_MODELS = [
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "zai-org/CogView3-Plus-3B",
    "PixArt-alpha/PixArt-XL-2-512x512",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
    "thu-ml/unidiffuser-v1",
    "kandinsky-community/kandinsky-3",
]

# Stable Cascade uses a special scheduler and architecture
SPECIAL_SCHEDULER_MODELS = [
    "stabilityai/stable-cascade",
]

# Enable unified scheduler (set to False to use model's original scheduler)
USE_UNIFIED_SCHEDULER = False

# Default generation parameters
DEFAULT_CONFIG = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,  # Classifier-free guidance scale for prompt adherence
    "width": 512,
    "height": 512,
    "seed": -1,  # -1 means random
    "negative_prompt": "",
    "use_unified_scheduler": False,  # Apply EulerDiscreteScheduler or FlowMatchEulerDiscreteScheduler
    "enable_profiling": True,  # Enable FLOPs/MACs computation profiling during inference
}

# Image size presets
IMAGE_SIZE_PRESETS = {
    "512x512": (512, 512),
    "768x768": (768, 768),
    "1024x1024": (1024, 1024),
    "1024x768 (Landscape)": (1024, 768),
    "768x1024 (Portrait)": (768, 1024),
    "1280x720 (16:9)": (1280, 720),
    "720x1280 (9:16)": (720, 1280),
}

# Output directory
OUTPUT_DIR = "outputs"

# Compute profiling configuration
ENABLE_COMPUTE_PROFILING = True  # Global flag to enable/disable FLOPs and MACs calculation

# Local model folder path
LOCAL_MODEL_DIR = "/data/liuzicheng/zhenyuan/models"

# Closed-source API model configurations
CLOSED_SOURCE_MODELS = {
    "openai": {
        "name": "OpenAI DALL-E",
        "short_name": "DALL-E 3",
        "models": ["dall-e-2", "dall-e-3"],
        "default_model": "dall-e-3",
        "supports_quality": True,
        "supports_style": True,
        "max_size": 1792,
        "api_key_env": "OPENAI_API_KEY",
    },
    "google": {
        "name": "Google Imagen",
        "short_name": "Imagen",
        "models": ["imagegeneration@005"],
        "default_model": "imagegeneration@005",
        "supports_quality": False,
        "supports_style": False,
        "max_size": 1536,
        "api_key_env": "GOOGLE_API_KEY",
        "additional_env": ["GOOGLE_PROJECT_ID"],
    },
    "bytedance": {
        "name": "Bytedance Cloud",
        "short_name": "Bytedance",
        "models": ["text2img-v1"],
        "default_model": "text2img-v1",
        "supports_quality": False,
        "supports_style": False,
        "max_size": 2048,
        "api_key_env": "BYTEDANCE_API_KEY",
    },
    "kling": {
        "name": "Kling AI",
        "short_name": "Kling",
        "models": ["kling-v1", "kling-v1-pro"],
        "default_model": "kling-v1",
        "supports_quality": False,
        "supports_style": False,
        "max_size": 2048,
        "api_key_env": "KLING_API_KEY",
    },
    "seeddream": {
        "name": "SeedDream 3.0",
        "short_name": "SeedDream",
        "models": ["high_aes_general_v30l_zt2i"],
        "default_model": "high_aes_general_v30l_zt2i",
        "supports_quality": False,
        "supports_style": False,
        "max_size": 2048,
        "api_key_env": "SEEDDREAM_ACCESS_KEY_ID",
        "additional_env": ["SEEDDREAM_SECRET_ACCESS_KEY"],
    }
}
