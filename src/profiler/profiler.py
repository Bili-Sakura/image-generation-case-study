"""
Compute profiler using calflops for accurate FLOPs/MACs calculation.
Uses direct pipeline method calls for reliable input capture.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from diffusers import DiffusionPipeline

from .model_reader import ModelReader
from .architecture_matcher import ArchitectureMatcher
from .component_profilers import MainModelProfiler, VAEProfiler, TextEncoderProfiler
from .utils import format_compute, format_params, to_gmacs, count_parameters


class ComputeProfiler:
    """
    Main profiler for measuring FLOPs, MACs, and inference time.
    Simple and clear: model files → inferred config → profiling
    """
    
    def __init__(self, enabled: bool = True, models_dir: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the compute profiler.
        
        Args:
            enabled: Whether to enable profiling
            models_dir: Path to models directory (default: project_root/models)
            config_path: Path to config.yaml fallback (default: use built-in)
        """
        self.enabled = enabled
        self._calflops_available = False
        
        if self.enabled:
            try:
                import calflops
                self._calflops_available = True
                self._calflops = calflops
                print("✓ Using calflops for profiling")
            except ImportError:
                print("⚠️  calflops not installed. Install with: pip install calflops")
                print("   FLOPs/MACs profiling will be disabled.")
                self.enabled = False
        
        # Initialize model reader and matcher
        self.model_reader = ModelReader(models_dir)
        self.matcher = ArchitectureMatcher(config_path)
        
        # Initialize component profilers
        if self._calflops_available:
            self.main_model_profiler = MainModelProfiler(self._calflops)
            self.vae_profiler = VAEProfiler(self._calflops)
            self.text_encoder_profiler = TextEncoderProfiler(self._calflops)
    
    def detect_architecture(
        self, 
        pipe: Optional[DiffusionPipeline] = None,
        model_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Detect architecture configuration.
        
        Strategy:
        1. If model_id provided, read from model directory (primary)
        2. If no model_id or not found, inspect pipeline (fallback)
        
        Args:
            pipe: Diffusion pipeline (optional if model_id provided)
            model_id: Model ID like "stabilityai/stable-diffusion-2-1-base"
        
        Returns:
            Complete architecture configuration or None
        """
        # Primary: Read from model directory
        if model_id:
            model_info = self.model_reader.read_model_info(model_id)
            if model_info:
                arch_config = self.matcher.build_architecture_config(model_info)
                return {
                    'name': model_id,
                    'source': 'model_directory',
                    'config': arch_config,
                    'detection_score': 10,
                }
        
        # Fallback: Inspect pipeline
        if pipe is not None:
            arch_config = self.matcher.match_from_pipeline(pipe)
            if arch_config:
                pipeline_name = pipe.__class__.__name__
                return {
                    'name': pipeline_name,
                    'source': 'pipeline_inspection',
                    'config': arch_config,
                    'detection_score': 5,
                }
        
        return None
    
    def profile_main_model(
        self,
        pipe: DiffusionPipeline,
        arch_info: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
        guidance_scale: float = 7.5,
        verbose: bool = False,
    ) -> Tuple[int, int]:
        """Profile main model (UNet/Transformer)."""
        if not self.enabled or not self._calflops_available:
            return 0, 0
        
        arch_config = arch_info['config']
        components = arch_config.get('components', {})
        main_model_attr = components.get('main_model_attr')
        
        main_model = getattr(pipe, main_model_attr, None)
        if main_model is None:
            return 0, 0
        
        main_model.eval()
        
        return self.main_model_profiler.profile(
            main_model,
            arch_config,
            pipeline_inputs,
            guidance_scale,
            verbose
        )
    
    def profile_vae(
        self,
        pipe: DiffusionPipeline,
        arch_info: Dict[str, Any],
        latents: torch.Tensor,
        verbose: bool = False,
    ) -> Tuple[int, int]:
        """Profile VAE decoder."""
        if not self.enabled or not self._calflops_available:
            return 0, 0
        
        arch_config = arch_info['config']
        components = arch_config.get('components')
        
        if not components.get('has_vae', False):
            return 0, 0
        
        if not hasattr(pipe, 'vae') or pipe.vae is None:
            return 0, 0
        
        return self.vae_profiler.profile(
            pipe.vae,
            latents,
            pipe,
            arch_config,
            verbose
        )
    
    def profile_text_encoders(
        self,
        pipe: DiffusionPipeline,
        arch_info: Dict[str, Any],
        tokenized_inputs_all: Dict[str, Dict[str, torch.Tensor]],
        verbose: bool = False,
    ) -> Dict[str, int]:
        """Profile all text encoders."""
        if not self.enabled or not self._calflops_available:
            return {
                "total_macs": 0,
                "total_params": 0,
                "text_encoder_1_macs": 0,
                "text_encoder_2_macs": 0,
                "text_encoder_3_macs": 0,
                "text_encoder_4_macs": 0,
                "text_encoder_1_params": 0,
                "text_encoder_2_params": 0,
                "text_encoder_3_params": 0,
                "text_encoder_4_params": 0,
                "max_token_length": 0,
            }
        
        arch_config = arch_info['config']
        
        return self.text_encoder_profiler.profile_all(
            pipe,
            tokenized_inputs_all,
            arch_config,
            verbose
        )
    
    def _capture_pipeline_inputs(
        self,
        pipe: DiffusionPipeline,
        arch_info: Dict[str, Any],
        height: int,
        width: int,
        prompt: str,
        guidance_scale: float,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Capture real inputs from a test inference run.
        
        Returns:
            Tuple of (pipeline_inputs, tokenized_inputs_all, latents)
            - pipeline_inputs: inputs for main model (transformer/unet)
            - tokenized_inputs_all: tokenized text inputs for text encoders
            - latents: final latents for VAE
        """
        # Use pipeline-specific input capture
        from .pipeline_specific_capture import PipelineSpecificCapture
        from .simplified_input_capture import SimplifiedInputCapture
        
        if verbose:
            print(f"  Using pipeline-specific input capture...")
        
        # Capture inputs using pipeline-specific methods
        captured_inputs = PipelineSpecificCapture.detect_and_capture(
            pipe=pipe,
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
        
        # Tokenize prompts
        tokenized_inputs_all = SimplifiedInputCapture.tokenize_prompts(
            pipe=pipe,
            prompt=prompt,
            negative_prompt="",
        )
        
        # Extract components
        pipeline_inputs = {k: v for k, v in captured_inputs.items() if k != 'vae_latents'}
        latents = captured_inputs.get('vae_latents')
        
        if verbose:
            print(f"  ✓ Captured inputs: {list(pipeline_inputs.keys())}")
            if latents is not None:
                print(f"  ✓ VAE latent shape: {latents.shape}")
        
        return pipeline_inputs, tokenized_inputs_all, latents
    
    def summarize(
        self,
        pipe: DiffusionPipeline,
        height: int = 512,
        width: int = 512,
        steps: int = 30,
        prompt: str = "a photo of a cat",
        guidance_scale: float = 7.5,
        model_id: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Summarize MACs for complete image generation.
        
        Args:
            pipe: The diffusion pipeline
            height: Image height
            width: Image width
            steps: Number of inference steps
            prompt: Text prompt
            guidance_scale: Guidance scale
            model_id: Optional model identifier
            verbose: Print verbose profiling information
        
        Returns:
            Dictionary with MAC counts and formatted strings
        """
        if not self.enabled or not self._calflops_available:
            return self._empty_summary(steps)
        
        # Detect architecture
        arch_info = self.detect_architecture(pipe, model_id)
        if arch_info is None:
            print(f"⚠️  Could not detect architecture for {model_id or 'pipeline'}")
            return self._empty_summary(steps)
        
        arch_name = arch_info['name']
        arch_config = arch_info['config']
        
        if verbose:
            print(f"\n📋 Detected Architecture: {arch_name}")
            print(f"   Source: {arch_info['source']}")
            print(f"   Description: {arch_config.get('description', 'N/A')}")
        
        # Capture real inputs from pipeline inference
        if verbose:
            print(f"\n🔍 Capturing pipeline inputs...")
        try:
            pipeline_inputs, tokenized_inputs_all, latents = self._capture_pipeline_inputs(
                pipe, arch_info, height, width, prompt, guidance_scale, verbose
            )
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to capture inputs: {e}")
                import traceback
                traceback.print_exc()
            return self._empty_summary(steps)
        
        # Profile each component
        if verbose:
            print(f"\n🔍 Profiling main model...")
        m_main_step, params_main = self.profile_main_model(
            pipe, arch_info, pipeline_inputs, guidance_scale, verbose
        )
        
        if verbose:
            print(f"\n🔍 Profiling VAE...")
        m_vae, params_vae = self.profile_vae(
            pipe, arch_info, latents, verbose
        )
        
        if verbose:
            print(f"\n🔍 Profiling text encoders...")
        text_results = self.profile_text_encoders(
            pipe, arch_info, tokenized_inputs_all, verbose
        )
        m_text_total = text_results["total_macs"]
        params_text_total = text_results["total_params"]
        max_token_length = text_results.get("max_token_length", 0)
        
        # Calculate totals
        total_macs = m_main_step * steps + m_vae + m_text_total
        total_params = params_main + params_vae + params_text_total
        
        # Build result dictionary
        result = {
            "enabled": True,
            "architecture": arch_name,
            "architecture_description": arch_config.get('description', ''),
            "detection_score": arch_info.get('detection_score', 0),
            "detection_source": arch_info.get('source', 'unknown'),
            
            # Profiling metadata
            "profiling_metadata": {
                "height": height,
                "width": width,
                "steps": steps,
                "prompt_token_length": max_token_length,
                "guidance_scale": guidance_scale,
            },
            
            # Main model
            "main_model_per_step_macs": m_main_step,
            "main_model_per_step_gmacs": round(to_gmacs(m_main_step), 3),
            "main_model_params": params_main,
            "main_model_params_str": format_params(params_main),
            
            # VAE
            "vae_macs": m_vae,
            "vae_gmacs": round(to_gmacs(m_vae), 3),
            "vae_params": params_vae,
            "vae_params_str": format_params(params_vae),
            
            # Text encoders
            "text_encoder_total_macs": m_text_total,
            "text_encoder_total_gmacs": round(to_gmacs(m_text_total), 3),
            "text_encoder_total_params": params_text_total,
            "text_encoder_total_params_str": format_params(params_text_total),
            "text_encoder_1_macs": text_results["text_encoder_1_macs"],
            "text_encoder_1_gmacs": round(to_gmacs(text_results["text_encoder_1_macs"]), 3),
            "text_encoder_1_params": text_results["text_encoder_1_params"],
            "text_encoder_1_params_str": format_params(text_results["text_encoder_1_params"]),
            "text_encoder_2_macs": text_results["text_encoder_2_macs"],
            "text_encoder_2_gmacs": round(to_gmacs(text_results["text_encoder_2_macs"]), 3),
            "text_encoder_2_params": text_results["text_encoder_2_params"],
            "text_encoder_2_params_str": format_params(text_results["text_encoder_2_params"]),
            "text_encoder_3_macs": text_results["text_encoder_3_macs"],
            "text_encoder_3_gmacs": round(to_gmacs(text_results["text_encoder_3_macs"]), 3),
            "text_encoder_3_params": text_results["text_encoder_3_params"],
            "text_encoder_3_params_str": format_params(text_results["text_encoder_3_params"]),
            "text_encoder_4_macs": text_results["text_encoder_4_macs"],
            "text_encoder_4_gmacs": round(to_gmacs(text_results["text_encoder_4_macs"]), 3),
            "text_encoder_4_params": text_results["text_encoder_4_params"],
            "text_encoder_4_params_str": format_params(text_results["text_encoder_4_params"]),
            
            # Totals
            f"total_{steps}_steps_macs": total_macs,
            f"total_{steps}_steps_gmacs": round(to_gmacs(total_macs), 3),
            "total_params": total_params,
            "total_params_str": format_params(total_params),
        }
        
        return result
    
    def _empty_summary(self, steps: int = 50) -> Dict[str, Any]:
        """Return empty summary when profiling is disabled or fails."""
        return {
            "enabled": False,
            "architecture": "unknown",
            "main_model_per_step_gmacs": 0.0,
            "text_encoder_total_gmacs": 0.0,
            "vae_gmacs": 0.0,
            f"total_{steps}_steps_gmacs": 0.0,
        }
    
    def print_summary(self, summary: Dict[str, Any]):
        """Pretty print profiling summary."""
        if not summary.get("enabled", False):
            print("⚠️  Profiling is disabled or unavailable")
            return
        
        print("\n" + "="*60)
        print(f"  Compute Profiling Summary")
        print("="*60)
        print(f"Architecture: {summary.get('architecture', 'unknown')}")
        if summary.get('architecture_description'):
            print(f"Description: {summary['architecture_description']}")
        if summary.get('detection_source'):
            print(f"Source: {summary['detection_source']}")
        
        # Print profiling metadata
        if 'profiling_metadata' in summary:
            metadata = summary['profiling_metadata']
            print(f"\nProfiling Configuration:")
            print(f"  Resolution: {metadata.get('height', 'N/A')}x{metadata.get('width', 'N/A')}")
            print(f"  Steps: {metadata.get('steps', 'N/A')}")
            print(f"  Prompt Token Length: {metadata.get('prompt_token_length', 'N/A')}")
            print(f"  Guidance Scale: {metadata.get('guidance_scale', 'N/A')}")
        
        print("-"*60)
        
        # Main model
        print(f"Main Model (per step):")
        print(f"  MACs: {format_compute(summary.get('main_model_per_step_macs', 0), 'MACs')}")
        print(f"  Params: {summary.get('main_model_params_str', '0')}")
        
        # Text encoders
        if summary.get('text_encoder_total_gmacs', 0) > 0:
            print(f"\nText Encoders (total):")
            print(f"  MACs: {format_compute(summary.get('text_encoder_total_macs', 0), 'MACs')}")
            print(f"  Params: {summary.get('text_encoder_total_params_str', '0')}")
            if summary.get('text_encoder_1_gmacs', 0) > 0:
                print(f"  - Encoder 1: {summary['text_encoder_1_gmacs']:.3f} GMACs, {summary.get('text_encoder_1_params_str', '0')}")
            if summary.get('text_encoder_2_gmacs', 0) > 0:
                print(f"  - Encoder 2: {summary['text_encoder_2_gmacs']:.3f} GMACs, {summary.get('text_encoder_2_params_str', '0')}")
            if summary.get('text_encoder_3_gmacs', 0) > 0:
                print(f"  - Encoder 3: {summary['text_encoder_3_gmacs']:.3f} GMACs, {summary.get('text_encoder_3_params_str', '0')}")
            if summary.get('text_encoder_4_gmacs', 0) > 0:
                print(f"  - Encoder 4: {summary['text_encoder_4_gmacs']:.3f} GMACs, {summary.get('text_encoder_4_params_str', '0')}")
        
        # VAE
        if summary.get('vae_gmacs', 0) > 0:
            print(f"\nVAE Decoder:")
            print(f"  MACs: {format_compute(summary.get('vae_macs', 0), 'MACs')}")
            print(f"  Params: {summary.get('vae_params_str', '0')}")
        
        # Total
        print("-"*60)
        total_key = [k for k in summary.keys() if k.startswith('total_') and k.endswith('_steps_gmacs')]
        if total_key:
            steps_str = total_key[0].replace('total_', '').replace('_steps_gmacs', '')
            print(f"Total ({steps_str} steps):")
            print(f"  MACs: {format_compute(summary.get(total_key[0].replace('gmacs', 'macs'), 0), 'MACs')}")
            print(f"  Total Params: {summary.get('total_params_str', '0')}")
        print("="*60 + "\n")
    
    def list_models(self) -> list:
        """List all available models."""
        return self.model_reader.list_all_models()


def create_profiler(enabled: bool = True, models_dir: Optional[str] = None, config_path: Optional[str] = None) -> ComputeProfiler:
    """
    Factory function to create a compute profiler.
    
    Args:
        enabled: Whether to enable profiling
        models_dir: Path to models directory
        config_path: Path to config.yaml fallback
    
    Returns:
        ComputeProfiler instance
    """
    return ComputeProfiler(enabled=enabled, models_dir=models_dir, config_path=config_path)

