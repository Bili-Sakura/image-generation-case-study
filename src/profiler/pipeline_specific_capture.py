"""
Pipeline-specific input capture methods.
Each pipeline type has its own unique handling based on diffusers implementation.
"""

import torch
from typing import Dict, Any
import math


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """
    Calculate timestep shift value for FLUX scheduler.
    Based on diffusers' FLUX pipeline implementation.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class PipelineSpecificCapture:
    """Capture inputs for specific pipeline types."""
    
    @staticmethod
    def capture_flux(pipe, prompt: str, height: int, width: int, guidance_scale: float) -> Dict[str, Any]:
        """FLUX.1 pipeline - FluxPipeline"""
        # FLUX uses a unique encode_prompt signature
        print(f"DEBUG: Calling encode_prompt...")
        encode_result = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe.device,
            num_images_per_prompt=1,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            max_sequence_length=512,
        )
        print(f"DEBUG: encode_prompt returned type: {type(encode_result)}, len: {len(encode_result) if isinstance(encode_result, tuple) else 'N/A'}")
        (prompt_embeds, pooled_prompt_embeds, text_ids) = encode_result
        
        latent_channels = pipe.transformer.config.in_channels // 4
        
        print(f"DEBUG: Calling prepare_latents...")
        latents_result = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=pipe.transformer.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        print(f"DEBUG: prepare_latents returned type: {type(latents_result)}, len: {len(latents_result) if isinstance(latents_result, tuple) else 'N/A'}")
        latents, latent_image_ids = latents_result
        
        # FLUX scheduler: Calculate mu for dynamic shifting or disable it
        if hasattr(pipe.scheduler, 'config') and hasattr(pipe.scheduler.config, 'use_dynamic_shifting'):
            if pipe.scheduler.config.use_dynamic_shifting:
                # For packed latents, sequence length is the second dimension
                image_seq_len = latents.shape[1]
                mu = calculate_shift(
                    image_seq_len,
                    pipe.scheduler.config.base_image_seq_len,
                    pipe.scheduler.config.max_image_seq_len,
                    pipe.scheduler.config.base_shift,
                    pipe.scheduler.config.max_shift,
                )
                pipe.scheduler.set_timesteps(1, device=pipe.device, mu=mu)
            else:
                pipe.scheduler.set_timesteps(1, device=pipe.device)
        else:
            pipe.scheduler.set_timesteps(1, device=pipe.device)
        
        timesteps = pipe.scheduler.timesteps
        
        guidance = torch.tensor([guidance_scale], device=pipe.device, dtype=pipe.transformer.dtype)
        
        # Unpack latents for VAE
        latents_for_vae = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents_for_vae = (latents_for_vae / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        
        return {
            'sample': latents,
            'hidden_states': latents,
            'timestep': timesteps[0] / 1000,
            'encoder_hidden_states': prompt_embeds,
            'pooled_projections': pooled_prompt_embeds,
            'txt_ids': text_ids,
            'img_ids': latent_image_ids,
            'guidance': guidance,
            'joint_attention_kwargs': None,
            'vae_latents': latents_for_vae,
        }
    
    @staticmethod
    def capture_sd3(pipe, prompt: str, height: int, width: int, guidance_scale: float) -> Dict[str, Any]:
        """Stable Diffusion 3 - StableDiffusion3Pipeline"""
        # SD3 requires prompt_2 and prompt_3
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,  # Can use same prompt
            prompt_3=None,  # Can use same prompt
            negative_prompt="",
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
        )
        
        result = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.transformer.config.in_channels,
            height=height,
            width=width,
            dtype=pipe.transformer.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        latents = result[0] if isinstance(result, tuple) else result
        
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        # SD3 expects timestep as 1d tensor, not scalar
        timestep_tensor = timesteps.reshape(-1) if len(timesteps.shape) == 0 else timesteps
        
        return {
            'sample': latents,
            'hidden_states': latents,
            'timestep': timestep_tensor,  # 1d array for SD3
            'encoder_hidden_states': prompt_embeds,
            'pooled_projections': pooled_prompt_embeds,
            'joint_attention_kwargs': None,
            'vae_latents': latents,
        }
    
    @staticmethod
    def capture_sdxl(pipe, prompt: str, height: int, width: int, guidance_scale: float) -> Dict[str, Any]:
        """Stable Diffusion XL - StableDiffusionXLPipeline"""
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt="",
        )
        
        result = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            height=height,
            width=width,
            dtype=pipe.unet.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        latents = result[0] if isinstance(result, tuple) else result
        
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = pipe._get_add_time_ids(
            (height, width), (0, 0), (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(device=pipe.device)
        
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        
        return {
            'sample': latents,
            'timestep': timesteps[0],
            'encoder_hidden_states': prompt_embeds,
            'added_cond_kwargs': added_cond_kwargs,
            'vae_latents': latents,
        }
    
    @staticmethod
    def capture_sd_v2(pipe, prompt: str, height: int, width: int, guidance_scale: float) -> Dict[str, Any]:
        """Stable Diffusion v1/v2 - StableDiffusionPipeline"""
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt="",
        )
        
        result = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            height=height,
            width=width,
            dtype=pipe.unet.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        latents = result[0] if isinstance(result, tuple) else result
        
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        return {
            'sample': latents,
            'timestep': timesteps[0],
            'encoder_hidden_states': prompt_embeds,
            'vae_latents': latents,
        }
    
    @staticmethod
    def capture_generic_transformer(pipe, prompt: str, height: int, width: int, guidance_scale: float) -> Dict[str, Any]:
        """
        Generic transformer-based pipeline (Lumina, PixArt, Sana, etc.)
        Tries different encode_prompt signatures automatically.
        """
        # Get main model
        if hasattr(pipe, 'transformer'):
            main_model = pipe.transformer
        elif hasattr(pipe, 'unet'):
            main_model = pipe.unet
        else:
            raise ValueError("No transformer or unet found")
        
        # Try different encode_prompt signatures
        prompt_embeds = None
        try:
            # Try most common signature
            result = pipe.encode_prompt(
                prompt=prompt,
                device=pipe.device,
                num_images_per_prompt=1,
            )
            # Handle tuple or single return
            prompt_embeds = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            # Try simpler signature
            try:
                result = pipe.encode_prompt(prompt, device=pipe.device)
                prompt_embeds = result[0] if isinstance(result, tuple) else result
            except:
                raise ValueError(f"Could not encode prompt: {e}")
        
        # Prepare latents
        result = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=main_model.config.in_channels,
            height=height,
            width=width,
            dtype=main_model.dtype,
            device=pipe.device,
            generator=None,
            latents=None,
        )
        latents = result[0] if isinstance(result, tuple) else result
        
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        return {
            'sample': latents,
            'timestep': timesteps[0],
            'encoder_hidden_states': prompt_embeds,
            'vae_latents': latents,
        }
    
    @staticmethod
    def capture_lumina_next(pipe, prompt: str, height: int, width: int, guidance_scale: float) -> Dict[str, Any]:
        """Lumina Next SFT - LuminaNextSFTT2IPipeline"""
        
        # 1. Encode prompt
        prompt_embeds, encoder_mask = pipe.encode_prompt(
            prompt,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
        )

        # 2. Prepare latents - detect main model type
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            main_model = pipe.transformer
        elif hasattr(pipe, 'unet') and pipe.unet is not None:
            main_model = pipe.unet
        else:
            print("DEBUG: dir(pipe):", dir(pipe))
            raise ValueError("Could not find transformer or unet in pipeline")
        
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.transformer.config.in_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=pipe.device,
            generator=None,
        )
        
        # 3. Get timestep
        pipe.scheduler.set_timesteps(1, device=pipe.device)
        timesteps = pipe.scheduler.timesteps

        # 4. Prepare image rotary embeddings
        image_rotary_emb = pipe.prepare_image_rotary_embeddings(latents)

        return {
            'sample': latents,
            'timestep': timesteps[0],
            'encoder_hidden_states': prompt_embeds,
            'encoder_mask': encoder_mask,
            'image_rotary_emb': image_rotary_emb,
            'vae_latents': latents,
        }

    @staticmethod
    def detect_and_capture(pipe, prompt: str, height: int, width: int, guidance_scale: float = 7.5) -> Dict[str, Any]:
        """
        Detect pipeline type and use appropriate capture method.
        """
        pipeline_class = pipe.__class__.__name__
        
        # Specific pipeline type detection
        if 'Flux' in pipeline_class:
            return PipelineSpecificCapture.capture_flux(pipe, prompt, height, width, guidance_scale)
        elif 'StableDiffusion3' in pipeline_class or 'SD3' in pipeline_class:
            return PipelineSpecificCapture.capture_sd3(pipe, prompt, height, width, guidance_scale)
        elif 'StableDiffusionXL' in pipeline_class:
            return PipelineSpecificCapture.capture_sdxl(pipe, prompt, height, width, guidance_scale)
        elif 'QwenImagePipeline' in pipeline_class:
            return PipelineSpecificCapture.capture_generic_transformer(pipe, prompt, height, width, guidance_scale)
        elif pipeline_class == 'StableDiffusionPipeline':
            return PipelineSpecificCapture.capture_sd_v2(pipe, prompt, height, width, guidance_scale)
        else:
            # Generic approach for other pipelines
            return PipelineSpecificCapture.capture_generic_transformer(pipe, prompt, height, width, guidance_scale)

