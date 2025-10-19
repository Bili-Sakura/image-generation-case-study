"""
Simplified input capture using direct pipeline method calls.
This is more reliable than using hooks and follows diffusers' design patterns.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from diffusers import DiffusionPipeline


class SimplifiedInputCapture:
    """
    Capture profiling inputs by directly calling pipeline preparation methods.
    This avoids the fragility of hooks and uses official pipeline APIs.
    """
    
    @staticmethod
    def capture_flux_inputs(
        pipe,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float = 3.5,
    ) -> Dict[str, Any]:
        """Capture inputs for FLUX transformer profiling."""
        
        # 1. Encode prompt using pipeline's encode_prompt method
        # FLUX doesn't use do_classifier_free_guidance parameter
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe.device,
            num_images_per_prompt=1,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            max_sequence_length=512,
        )
        
        # 2. Prepare latents using pipeline's prepare_latents method
        latent_channels = pipe.transformer.config.in_channels // 4
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=pipe.transformer.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        
        # 3. Get timestep from scheduler
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        # 4. Prepare latent image ids (FLUX-specific)
        latent_image_ids = pipe._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            pipe.device,
            pipe.transformer.dtype,
        )
        
        # 5. Prepare guidance
        guidance = torch.tensor([guidance_scale], device=pipe.device, dtype=pipe.transformer.dtype)
        
        # 6. Get VAE latents by unpacking
        packed_latents = latents
        latent_image_ids_for_vae = latent_image_ids
        
        # Unpack for VAE
        latents_for_vae = pipe._unpack_latents(
            packed_latents,
            height,
            width,
            pipe.vae_scale_factor,
        )
        latents_for_vae = (latents_for_vae / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        
        return {
            # Main model inputs
            'sample': packed_latents,
            'hidden_states': packed_latents,
            'timestep': timesteps[0] / 1000,  # FLUX uses normalized timesteps
            'encoder_hidden_states': prompt_embeds,
            'pooled_projections': pooled_prompt_embeds,
            'txt_ids': text_ids,
            'img_ids': latent_image_ids,
            'guidance': guidance,
            'joint_attention_kwargs': None,
            # VAE inputs
            'vae_latents': latents_for_vae,
            # Text encoder inputs (already have the tokenized versions)
            'tokenized_prompt': prompt,
        }
    
    @staticmethod
    def capture_sd3_inputs(
        pipe,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """Capture inputs for SD3 transformer profiling."""
        
        # 1. Encode prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt="",
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
        )
        
        # 2. Prepare latents
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.transformer.config.in_channels,
            height=height,
            width=width,
            dtype=pipe.transformer.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        
        # 3. Get timestep
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        return {
            'sample': latents,
            'hidden_states': latents,
            'timestep': timesteps[0],
            'encoder_hidden_states': prompt_embeds,
            'pooled_projections': pooled_prompt_embeds,
            'joint_attention_kwargs': None,
            # VAE inputs
            'vae_latents': latents,
        }
    
    @staticmethod
    def capture_sdxl_inputs(
        pipe,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """Capture inputs for SDXL UNet profiling."""
        
        # 1. Encode prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt="",
        )
        
        # 2. Prepare latents
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            height=height,
            width=width,
            dtype=pipe.unet.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        
        # 3. Get timestep
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        # 4. Prepare added conditioning
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(device=pipe.device)
        
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids
        }
        
        return {
            'sample': latents,
            'timestep': timesteps[0],
            'encoder_hidden_states': prompt_embeds,
            'added_cond_kwargs': added_cond_kwargs,
            # VAE inputs
            'vae_latents': latents,
        }
    
    @staticmethod
    def capture_sd2_inputs(
        pipe,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """Capture inputs for SD 2.x and similar pipelines (generic approach)."""
        
        # 1. Encode prompt - handle different return signatures
        try:
            # Try with do_classifier_free_guidance parameter
            result = pipe.encode_prompt(
                prompt=prompt,
                device=pipe.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=guidance_scale > 1.0,
                negative_prompt="",
            )
        except TypeError:
            # Fallback: try without do_classifier_free_guidance
            try:
                result = pipe.encode_prompt(
                    prompt=prompt,
                    device=pipe.device,
                    num_images_per_prompt=1,
                    negative_prompt="",
                )
            except TypeError:
                # Last fallback: minimal args
                result = pipe.encode_prompt(
                    prompt=prompt,
                    device=pipe.device,
                    num_images_per_prompt=1,
                )
        
        # Handle different return types
        if isinstance(result, tuple):
            # Could be (prompt_embeds, negative_embeds) or more values
            prompt_embeds = result[0]
        else:
            prompt_embeds = result
        
        # 2. Prepare latents - detect main model type
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            main_model = pipe.transformer
        elif hasattr(pipe, 'unet') and pipe.unet is not None:
            main_model = pipe.unet
        else:
            raise ValueError("Could not find transformer or unet in pipeline")
        
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=main_model.config.in_channels,
            height=height,
            width=width,
            dtype=main_model.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        
        # 3. Get timestep
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        return {
            'sample': latents,
            'timestep': timesteps[0],
            'encoder_hidden_states': prompt_embeds,
            # VAE inputs
            'vae_latents': latents,
        }
    
    @staticmethod
    def capture_inputs(
        pipe: DiffusionPipeline,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float = 7.5,
        architecture_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Auto-detect pipeline type and capture appropriate inputs.
        
        Args:
            pipe: Diffusion pipeline
            prompt: Text prompt
            height: Image height
            width: Image width
            guidance_scale: Guidance scale
            architecture_hint: Optional hint about architecture (e.g., 'flux', 'sd3')
        
        Returns:
            Dictionary of captured inputs ready for profiling
        """
        
        # Get pipeline class name for detection
        pipeline_class = pipe.__class__.__name__
        
        # Detect architecture based on pipeline class name
        # Priority: explicit hint > class name detection
        if architecture_hint:
            arch_lower = architecture_hint.lower()
            # Map architecture hints to capture methods
            if 'flux' in arch_lower:
                return SimplifiedInputCapture.capture_flux_inputs(pipe, prompt, height, width, guidance_scale)
            elif 'sd3' in arch_lower or 'stable-diffusion-3' in arch_lower:
                return SimplifiedInputCapture.capture_sd3_inputs(pipe, prompt, height, width, guidance_scale)
            elif 'sdxl' in arch_lower or 'xl' in arch_lower:
                return SimplifiedInputCapture.capture_sdxl_inputs(pipe, prompt, height, width, guidance_scale)
        
        # Auto-detect from pipeline class name
        if 'Flux' in pipeline_class:
            return SimplifiedInputCapture.capture_flux_inputs(pipe, prompt, height, width, guidance_scale)
        elif 'StableDiffusion3' in pipeline_class or 'SD3' in pipeline_class:
            return SimplifiedInputCapture.capture_sd3_inputs(pipe, prompt, height, width, guidance_scale)
        elif 'StableDiffusionXL' in pipeline_class or 'SDXL' in pipeline_class:
            return SimplifiedInputCapture.capture_sdxl_inputs(pipe, prompt, height, width, guidance_scale)
        elif 'StableDiffusion' in pipeline_class:
            # Generic SD (v1, v2, etc.)
            return SimplifiedInputCapture.capture_sd2_inputs(pipe, prompt, height, width, guidance_scale)
        
        # Fallback: Try generic SD2 approach (most common)
        # This works for many pipelines: Lumina, PixArt, Sana, etc.
        return SimplifiedInputCapture.capture_sd2_inputs(pipe, prompt, height, width, guidance_scale)
    
    @staticmethod
    def tokenize_prompts(
        pipe: DiffusionPipeline,
        prompt: str,
        negative_prompt: str = "",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Tokenize prompts for all text encoders."""
        tokenized_inputs_all = {}
        
        # Handle different tokenizer scenarios
        tokenizers = []
        if hasattr(pipe, 'tokenizer') and pipe.tokenizer is not None:
            tokenizers.append(('text_encoder_1', pipe.tokenizer))
        if hasattr(pipe, 'tokenizer_2') and pipe.tokenizer_2 is not None:
            tokenizers.append(('text_encoder_2', pipe.tokenizer_2))
        if hasattr(pipe, 'tokenizer_3') and pipe.tokenizer_3 is not None:
            tokenizers.append(('text_encoder_3', pipe.tokenizer_3))
        
        for encoder_name, tokenizer in tokenizers:
            # Handle potential overflow for max_length
            max_length = tokenizer.model_max_length
            if max_length > 2**31 - 1:  # Check if it's a very large number
                max_length = 2048  # Fallback to a reasonable value
            
            pos_tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            neg_tokens = tokenizer(
                negative_prompt if negative_prompt else "",
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_inputs_all[encoder_name] = {
                'positive': pos_tokens,
                'negative': neg_tokens,
            }
        
        return tokenized_inputs_all

