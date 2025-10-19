"""
Model reader - extracts all necessary information from model directories.
Primary source: model JSON files (model_index.json, config.json, etc.)
Fallback: config.yaml for architecture patterns
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class ModelReader:
    """
    Reads model information from model directory JSON files.
    Infers profiling configuration from model structure.
    """
    
    def __init__(self, models_base_dir: Optional[str] = None):
        """
        Initialize model reader.
        
        Args:
            models_base_dir: Path to models directory (default: project_root/models)
        """
        if models_base_dir is None:
            models_base_dir = Path(__file__).parent.parent.parent / "models"
        else:
            models_base_dir = Path(models_base_dir)
        
        self.models_base_dir = models_base_dir
    
    def read_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Read complete model information from model directory.
        
        Args:
            model_id: Model ID like "stabilityai/stable-diffusion-2-1-base"
        
        Returns:
            Dictionary with all model info, or None if not found
        """
        # Parse model_id
        parts = model_id.split('/')
        if len(parts) != 2:
            return None
        
        org, model_name = parts
        model_dir = self.models_base_dir / org / model_name
        
        if not model_dir.exists():
            return None
        
        # Read model_index.json
        model_index_path = model_dir / "model_index.json"
        if not model_index_path.exists():
            return None
        
        try:
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
        except Exception as e:
            print(f"⚠️  Error reading model_index.json for {model_id}: {e}")
            return None
        
        # Extract basic info
        pipeline_class = model_index.get('_class_name')
        if not pipeline_class:
            return None
        
        # Detect main model type and read its config
        main_model_info = self._read_main_model_info(model_dir, model_index)
        if not main_model_info:
            return None
        
        # Detect components
        components = self._detect_components(model_index)
        
        # Read component configs for additional info
        vae_info = self._read_vae_info(model_dir) if components['has_vae'] else {}
        text_encoder_info = self._read_text_encoder_info(model_dir, components['text_encoder_count'])
        tokenizer_info = self._read_tokenizer_info(model_dir, components['text_encoder_count'])
        
        # Build complete model info
        model_info = {
            'model_id': model_id,
            'model_dir': str(model_dir),
            'pipeline_class': pipeline_class,
            'main_model_type': main_model_info['type'],
            'main_model_class': main_model_info['class'],
            'main_model_attr': main_model_info['attr'],
            'components': components,
            'dimensions': main_model_info.get('dimensions', {}),
            'vae_info': vae_info,
            'text_encoder_info': text_encoder_info,
            'tokenizer_info': tokenizer_info,
        }
        
        return model_info
    
    def _read_main_model_info(self, model_dir: Path, model_index: dict) -> Optional[Dict[str, Any]]:
        """Read main model (transformer or unet) information."""
        # Check for transformer
        if 'transformer' in model_index:
            config_path = model_dir / "transformer" / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    return {
                        'type': 'transformer',
                        'attr': 'transformer',
                        'class': config.get('_class_name', model_index['transformer'][1]),
                        'dimensions': {
                            'in_channels': config.get('in_channels'),
                            'joint_attention_dim': config.get('joint_attention_dim'),
                            'caption_projection_dim': config.get('caption_projection_dim'),
                            'pooled_projection_dim': config.get('pooled_projection_dim'),
                            'num_attention_heads': config.get('num_attention_heads'),
                            'num_layers': config.get('num_layers'),
                            'patch_size': config.get('patch_size'),
                        }
                    }
                except Exception:
                    pass
        
        # Check for unet
        if 'unet' in model_index:
            config_path = model_dir / "unet" / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    return {
                        'type': 'unet',
                        'attr': 'unet',
                        'class': config.get('_class_name', model_index['unet'][1]),
                        'dimensions': {
                            'in_channels': config.get('in_channels'),
                            'out_channels': config.get('out_channels'),
                            'cross_attention_dim': config.get('cross_attention_dim'),
                            'attention_head_dim': config.get('attention_head_dim'),
                            'block_out_channels': config.get('block_out_channels'),
                        }
                    }
                except Exception:
                    pass
        
        return None
    
    def _detect_components(self, model_index: dict) -> Dict[str, Any]:
        """Detect what components the model has."""
        return {
            'has_unet': 'unet' in model_index,
            'has_transformer': 'transformer' in model_index,
            'has_vae': 'vae' in model_index,
            'has_movq': 'movq' in model_index,
            'text_encoder_count': sum([
                'text_encoder' in model_index,
                'text_encoder_2' in model_index,
                'text_encoder_3' in model_index,
                'text_encoder_4' in model_index,
            ]),
        }
    
    def _read_vae_info(self, model_dir: Path) -> Dict[str, Any]:
        """Read VAE configuration."""
        config_path = model_dir / "vae" / "config.json"
        if not config_path.exists():
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Calculate downscale factor from block_out_channels length
            block_out_channels = config.get('block_out_channels', [])
            downscale_factor = 2 ** len(block_out_channels) if block_out_channels else None
            
            return {
                'latent_channels': config.get('latent_channels'),
                'downscale_factor': downscale_factor,
                'scaling_factor': config.get('scaling_factor'),
            }
        except Exception:
            return {}
    
    def _read_text_encoder_info(self, model_dir: Path, text_encoder_count: int) -> Dict[str, Any]:
        """Read text encoder configurations."""
        info = {}
        
        for i in range(1, text_encoder_count + 1):
            encoder_name = 'text_encoder' if i == 1 else f'text_encoder_{i}'
            config_path = model_dir / encoder_name / "config.json"
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    info[encoder_name] = {
                        'hidden_size': config.get('hidden_size') or config.get('d_model'),
                        'max_position_embeddings': config.get('max_position_embeddings'),
                        'model_type': config.get('model_type'),
                    }
                except Exception:
                    pass
        
        return info
    
    def _read_tokenizer_info(self, model_dir: Path, text_encoder_count: int) -> Dict[str, Any]:
        """Read tokenizer configurations."""
        info = {}
        
        for i in range(1, text_encoder_count + 1):
            tokenizer_name = 'tokenizer' if i == 1 else f'tokenizer_{i}'
            config_path = model_dir / tokenizer_name / "tokenizer_config.json"
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    info[tokenizer_name] = {
                        'model_max_length': config.get('model_max_length'),
                        'tokenizer_class': config.get('tokenizer_class'),
                    }
                except Exception:
                    pass
        
        return info
    
    def list_all_models(self) -> list:
        """List all available models in the models directory."""
        if not self.models_base_dir.exists():
            return []
        
        models = []
        for org_dir in self.models_base_dir.iterdir():
            if not org_dir.is_dir() or org_dir.name.startswith('.'):
                continue
            
            for model_dir in org_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith('.'):
                    continue
                
                if (model_dir / "model_index.json").exists():
                    models.append(f"{org_dir.name}/{model_dir.name}")
        
        return sorted(models)

