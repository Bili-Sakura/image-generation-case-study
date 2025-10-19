"""
Architecture matcher - infers profiling configuration from model info.
Uses config.yaml only as fallback for unknown patterns.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ArchitectureMatcher:
    """
    Matches model info to profiling configuration.
    Infers settings where possible, falls back to config.yaml patterns.
    """
    
    # Mapping from model classes to wrapper classes
    WRAPPER_MAP = {
        'UNet2DConditionModel': 'UNetWrapper',
        'FluxTransformer2DModel': 'FluxTransformerWrapper',
        'SD3Transformer2DModel': 'SD3TransformerWrapper',
        'SanaTransformer2DModel': 'GenericTransformerWrapper',
        'PixArtTransformer2DModel': 'GenericTransformerWrapper',
        'Transformer2DModel': 'GenericTransformerWrapper',
        'CogView3PlusTransformer2DModel': 'GenericTransformerWrapper',
        'CogView4Transformer2DModel': 'GenericTransformerWrapper',
        'QwenImageTransformer2DModel': 'GenericTransformerWrapper',
        'HiDreamImageTransformer2DModel': 'GenericTransformerWrapper',
        'LuminaNextDiT2DModel': 'LuminaNextTransformerWrapper',
        'Lumina2Transformer2DModel': 'Lumina2TransformerWrapper',
        'HunyuanDiT2DModel': 'GenericTransformerWrapper',
        'Kandinsky3UNet': 'KandinskyUNetWrapper',
    }
    
    # Special input requirements for certain model classes
    SPECIAL_INPUTS = {
        'FluxTransformer2DModel': ['img_ids', 'txt_ids', 'guidance', 'pooled_projections'],
        'SD3Transformer2DModel': ['pooled_projections'],
        'LuminaNextDiT2DModel': ['encoder_mask', 'image_rotary_emb'],
        'Lumina2Transformer2DModel': ['encoder_attention_mask'],
        'Kandinsky3UNet': ['encoder_attention_mask'],
        'CogView3PlusTransformer2DModel': ['original_size', 'target_size', 'crop_coords'],
        'CogView4Transformer2DModel': ['original_size', 'target_size', 'crop_coords'],
    }
    
    # Pipeline-specific special inputs
    PIPELINE_SPECIAL_INPUTS = {
        'StableDiffusionXLPipeline': ['added_cond_kwargs'],
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize architecture matcher.
        
        Args:
            config_path: Path to config.yaml fallback (optional)
        """
        self.fallback_config = {}
        
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.fallback_config = yaml.safe_load(f)
    
    def build_architecture_config(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build complete architecture configuration from model info.
        All values are extracted from model JSON files - no hardcoded defaults.
        
        Args:
            model_info: Model information from ModelReader
        
        Returns:
            Complete architecture configuration for profiling
        """
        main_model_class = model_info['main_model_class']
        pipeline_class = model_info['pipeline_class']
        components = model_info['components']
        dimensions = model_info.get('dimensions', {})
        vae_info = model_info.get('vae_info', {})
        text_encoder_info = model_info.get('text_encoder_info', {})
        tokenizer_info = model_info.get('tokenizer_info', {})
        
        # Infer wrapper class from model class
        wrapper_class = self.WRAPPER_MAP.get(main_model_class, 'GenericTransformerWrapper')
        
        # Infer CFG multiplier (FLUX doesn't use CFG, others do)
        cfg_multiplier = 1 if 'Flux' in pipeline_class else 2
        
        # Get embedding dimension from transformer/unet config or text encoder config
        # Priority: joint_attention_dim > cross_attention_dim > text_encoder_2 hidden_size > text_encoder hidden_size
        embedding_dim = (
            dimensions.get('joint_attention_dim') or 
            dimensions.get('cross_attention_dim') or
            (text_encoder_info.get('text_encoder_2', {}).get('hidden_size')) or
            (text_encoder_info.get('text_encoder', {}).get('hidden_size'))
        )
        
        # Get latent channels
        # For FLUX: transformer's in_channels (64) - VAE outputs 16 channels but they get packed to 64
        # For other transformers: use in_channels from transformer config
        # For UNets: use VAE's latent_channels or in_channels
        if main_model_class == 'FluxTransformer2DModel':
            latent_channels = dimensions.get('in_channels')  # 64 for FLUX
        elif model_info['main_model_type'] == 'transformer':
            latent_channels = dimensions.get('in_channels')  # Direct from transformer config
        else:
            # For UNets, prefer VAE latent_channels
            latent_channels = vae_info.get('latent_channels') or dimensions.get('in_channels')
        
        # Get downscale factor from VAE/MOVQ config with intelligent fallback
        downscale_factor = vae_info.get('downscale_factor')
        if not downscale_factor:
            # Fallback: MOVQ models (Kandinsky) use 4x downscale, others use 8x
            if components['has_movq']:
                downscale_factor = 4
            else:
                downscale_factor = 8
        
        # Get text sequence length from tokenizer config with fallback
        # Priority: tokenizer_2 (for multi-encoder models like FLUX with T5) > tokenizer
        text_seq_len = (
            tokenizer_info.get('tokenizer_2', {}).get('model_max_length') or
            tokenizer_info.get('tokenizer', {}).get('model_max_length')
        )
        if not text_seq_len:
            # Fallback to text encoder's max_position_embeddings
            text_seq_len = (
                text_encoder_info.get('text_encoder_2', {}).get('max_position_embeddings') or
                text_encoder_info.get('text_encoder', {}).get('max_position_embeddings') or
                77  # Standard CLIP max length
            )
        
        # Get special inputs from model class and pipeline class
        special_inputs = self.SPECIAL_INPUTS.get(main_model_class, []).copy()
        pipeline_inputs = self.PIPELINE_SPECIAL_INPUTS.get(pipeline_class, [])
        special_inputs.extend(pipeline_inputs)
        
        # Validate critical values with better error messages
        if not embedding_dim:
            print(f"⚠️  Warning: Could not determine embedding_dim for {model_info['model_id']}, using fallback 768")
            embedding_dim = 768  # Common CLIP dimension
        if not latent_channels:
            print(f"⚠️  Warning: Could not determine latent_channels for {model_info['model_id']}, using fallback 4")
            latent_channels = 4  # Standard SD latent channels
        
        # Build complete config
        arch_config = {
            'description': f"{model_info['model_id']}",
            'detection': {
                'model_class': main_model_class,
                'pipeline_class': pipeline_class,
            },
            'components': {
                'main_model_type': model_info['main_model_type'],
                'main_model_attr': model_info['main_model_attr'],
                'text_encoder_count': components['text_encoder_count'],
                'has_vae': components['has_vae'],
                'has_movq': components['has_movq'],
                'latent_channels': latent_channels,
                'downscale_factor': downscale_factor,
            },
            'profiling': {
                'wrapper_class': wrapper_class,
                'cfg_multiplier': cfg_multiplier,
                'requires_sdpa_tracking': True,
                'text_seq_len': text_seq_len,
                'embedding_dim': embedding_dim,
            }
        }
        
        # Add special inputs if needed
        if special_inputs:
            arch_config['profiling']['special_inputs'] = special_inputs
        
        return arch_config
    
    def match_from_pipeline(self, pipe) -> Optional[Dict[str, Any]]:
        """
        Match architecture from pipeline inspection (fallback method).
        
        Args:
            pipe: DiffusionPipeline instance
        
        Returns:
            Architecture config or None
        """
        # Get pipeline class
        pipeline_class = pipe.__class__.__name__
        
        # Get main model
        main_model = None
        main_model_class = None
        main_model_type = None
        main_model_attr = None
        
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            main_model = pipe.transformer
            main_model_class = main_model.__class__.__name__
            main_model_type = 'transformer'
            main_model_attr = 'transformer'
        elif hasattr(pipe, 'unet') and pipe.unet is not None:
            main_model = pipe.unet
            main_model_class = main_model.__class__.__name__
            main_model_type = 'unet'
            main_model_attr = 'unet'
        
        if not main_model_class:
            return None
        
        # Check if we can find a match in fallback config
        architectures = self.fallback_config.get('architectures', {})
        for arch_name, arch_config in architectures.items():
            detection = arch_config.get('detection', {})
            if (detection.get('model_class') == main_model_class and 
                detection.get('pipeline_class') == pipeline_class):
                return arch_config
        
        # If no match, build minimal config
        components = {
            'has_vae': hasattr(pipe, 'vae') and pipe.vae is not None,
            'has_movq': hasattr(pipe, 'movq') and pipe.movq is not None,
            'text_encoder_count': sum([
                hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None,
                hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None,
                hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None,
                hasattr(pipe, 'text_encoder_4') and pipe.text_encoder_4 is not None,
            ]),
        }
        
        return {
            'description': f"Auto-detected {pipeline_class}",
            'detection': {
                'model_class': main_model_class,
                'pipeline_class': pipeline_class,
            },
            'components': {
                'main_model_type': main_model_type,
                'main_model_attr': main_model_attr,
                'text_encoder_count': components['text_encoder_count'],
                'has_vae': components['has_vae'],
                'has_movq': components['has_movq'],
                'latent_channels': 4,
                'downscale_factor': 8,
            },
            'profiling': {
                'wrapper_class': self.WRAPPER_MAP.get(main_model_class, 'GenericTransformerWrapper'),
                'cfg_multiplier': 1 if 'Flux' in pipeline_class else 2,
                'requires_sdpa_tracking': True,
                'text_seq_len': 77,
                'embedding_dim': 2048,
            }
        }

