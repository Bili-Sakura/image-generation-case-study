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
        "short_name": "CogView3",
        "requires_safety_checker": False,
        "pipeline_class": "CogView3PlusPipeline",
    },
    "PixArt-alpha/PixArt-XL-2-512x512": {
        "name": "PixArt-XL 2",
        "short_name": "PixArt-XL",
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
        "short_name": "SD3",
        "requires_safety_checker": False,
        "pipeline_class": "StableDiffusion3Pipeline",
    },
    "black-forest-labs/FLUX.1-dev": {
        "name": "FLUX.1 Dev",
        "short_name": "FLUX.1",
        "requires_safety_checker": False,
        "pipeline_class": "FluxPipeline",
    },
    "Efficient-Large-Model/Sana_600M_512px_diffusers": {
        "name": "Sana 600M 512px",
        "short_name": "Sana",
        "requires_safety_checker": False,
        "pipeline_class": "SanaPipeline",
    },
    "Qwen/Qwen-Image": {
        "name": "Qwen Image",
        "short_name": "Qwen",
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
        "short_name": "Cascade",
        "requires_safety_checker": False,
        "pipeline_class": "StableCascadeCombinedPipeline",
    },
    "zai-org/CogView4-6B": {
        "name": "CogView4 6B",
        "short_name": "CogView4",
        "requires_safety_checker": False,
        "pipeline_class": "DiffusionPipeline",
    },
}

# Default models to load on startup
DEFAULT_MODELS = [
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-cascade",
    "stabilityai/stable-diffusion-3-medium-diffusers",
]

# Default generation parameters
DEFAULT_CONFIG = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,  # Classifier-free guidance scale for prompt adherence
    "width": 512,
    "height": 512,
    "seed": -1,  # -1 means random
    "negative_prompt": "",
    "scheduler": "EulerDiscreteScheduler",  # Unified scheduler for all models
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

# Local model folder path
LOCAL_MODEL_DIR = "/data/liuzicheng/zhenyuan/models"
