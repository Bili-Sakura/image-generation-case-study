"""
Model wrappers for profiling with thop.
These wrappers make different architectures compatible with thop.profile().
"""

import torch
import torch.nn as nn
from typing import Optional


class UNetWrapper(nn.Module):
    """Wrapper for UNet models to make them compatible with profiling."""
    def __init__(self, unet, timestep, encoder_hidden_states, added_cond_kwargs=None):
        super().__init__()
        self.unet = unet
        self.timestep = timestep
        self.encoder_hidden_states = encoder_hidden_states
        self.added_cond_kwargs = added_cond_kwargs

    def forward(self, x, **kwargs):
        # Merge kwargs from profiler with a default empty dict
        final_kwargs = {**kwargs, **(self.added_cond_kwargs or {})}
        return self.unet(
            x,
            self.timestep,
            encoder_hidden_states=self.encoder_hidden_states,
            added_cond_kwargs=final_kwargs,
            return_dict=False
        )[0]


class FluxTransformerWrapper(nn.Module):
    """
    Wrapper for FLUX transformer to make it compatible with thop.profile().
    """
    def __init__(
        self, 
        transformer, 
        encoder_hidden_states, 
        pooled_projections, 
        timestep, 
        img_ids, 
        txt_ids, 
        guidance=None
    ):
        super().__init__()
        self.transformer = transformer
        self.encoder_hidden_states = encoder_hidden_states
        self.pooled_projections = pooled_projections
        self.timestep = timestep
        self.img_ids = img_ids
        self.txt_ids = txt_ids
        self.guidance = guidance
    
    def forward(self, hidden_states, **kwargs):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            pooled_projections=self.pooled_projections,
            timestep=self.timestep,
            img_ids=self.img_ids,
            txt_ids=self.txt_ids,
            guidance=self.guidance,
            return_dict=False
        )[0]


class SD3TransformerWrapper(nn.Module):
    """
    Wrapper for SD3 transformer to make it compatible with thop.profile().
    """
    def __init__(self, transformer, encoder_hidden_states, pooled_projections, timestep):
        super().__init__()
        self.transformer = transformer
        self.encoder_hidden_states = encoder_hidden_states
        self.pooled_projections = pooled_projections
        self.timestep = timestep
    
    def forward(self, hidden_states, **kwargs):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            pooled_projections=self.pooled_projections,
            timestep=self.timestep,
            return_dict=False
        )[0]


class GenericTransformerWrapper(nn.Module):
    """Generic wrapper for various transformer models."""
    def __init__(self, transformer, **kwargs):
        super().__init__()
        self.transformer = transformer
        self.kwargs = kwargs

    def forward(self, hidden_states):
        # The first argument from calflops is the dummy tensor.
        # Pass the real arguments from self.kwargs.
        # Ensure hidden_states is passed correctly, and other kwargs follow.
        return self.transformer(
            hidden_states,
            encoder_hidden_states=self.kwargs.get('encoder_hidden_states'),
            timestep=self.kwargs.get('timestep'),
            return_dict=False
        )[0]


class LuminaNextTransformerWrapper(nn.Module):
    """Wrapper for Lumina-Next."""
    def __init__(self, transformer, **kwargs):
        super().__init__()
        self.transformer = transformer
        self.kwargs = kwargs

    def forward(self, hidden_states, **kwargs):
        final_kwargs = {**self.kwargs, **kwargs}
        return self.transformer(
            hidden_states,
            **final_kwargs,
            return_dict=False
        )[0]


class Lumina2TransformerWrapper(nn.Module):
    """
    Wrapper for Lumina2 transformer (needs encoder_attention_mask).
    """
    def __init__(self, transformer, encoder_hidden_states, timestep, encoder_attention_mask):
        super().__init__()
        self.transformer = transformer
        self.encoder_hidden_states = encoder_hidden_states
        self.timestep = timestep
        self.encoder_attention_mask = encoder_attention_mask
    
    def forward(self, hidden_states):
        return self.transformer(
            hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_attention_mask,
            return_dict=False
        )[0]


class KandinskyUNetWrapper(nn.Module):
    """
    Wrapper for Kandinsky-3 UNet (needs encoder_attention_mask).
    """
    def __init__(self, unet, timestep, encoder_hidden_states, encoder_attention_mask):
        super().__init__()
        self.unet = unet
        self.timestep = timestep
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_attention_mask = encoder_attention_mask
    
    def forward(self, sample):
        return self.unet(
            sample,
            self.timestep,
            encoder_hidden_states=self.encoder_hidden_states,
            encoder_attention_mask=self.encoder_attention_mask,
            return_dict=False
        )[0]


class Kandinsky2UNetWrapper(nn.Module):
    """
    Wrapper for Kandinsky-2.1 UNet (simpler than Kandinsky-3).
    """
    def __init__(self, unet, timestep, encoder_hidden_states):
        super().__init__()
        self.unet = unet
        self.timestep = timestep
        self.encoder_hidden_states = encoder_hidden_states
    
    def forward(self, sample):
        return self.unet(
            sample,
            self.timestep,
            encoder_hidden_states=self.encoder_hidden_states,
            return_dict=False
        )[0]


class VAEWrapper(nn.Module):
    """
    Wrapper for VAE decoder to make it compatible with thop.profile().
    """
    def __init__(self, vae, scaling_factor: float):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor
    
    def forward(self, z):
        return self.vae.decode(z / self.scaling_factor, return_dict=False)[0]


class TextEncoderWrapper(nn.Module):
    """
    Wrapper for text encoder to make it compatible with thop.profile().
    """
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
    
    def forward(self, input_ids):
        return self.text_encoder(input_ids, return_dict=False)[0]


# Wrapper registry for easy lookup
WRAPPER_REGISTRY = {
    "UNetWrapper": UNetWrapper,
    "FluxTransformerWrapper": FluxTransformerWrapper,
    "SD3TransformerWrapper": SD3TransformerWrapper,
    "GenericTransformerWrapper": GenericTransformerWrapper,
    "LuminaNextTransformerWrapper": LuminaNextTransformerWrapper,
    "Lumina2TransformerWrapper": Lumina2TransformerWrapper,
    "KandinskyUNetWrapper": KandinskyUNetWrapper,
    "Kandinsky2UNetWrapper": Kandinsky2UNetWrapper,
    "VAEWrapper": VAEWrapper,
    "TextEncoderWrapper": TextEncoderWrapper,
}


def get_wrapper_class(wrapper_name: str):
    """
    Get wrapper class by name.
    
    Args:
        wrapper_name: Name of the wrapper class
    
    Returns:
        Wrapper class
    
    Raises:
        KeyError: If wrapper name is not found
    """
    if wrapper_name not in WRAPPER_REGISTRY:
        raise KeyError(f"Wrapper '{wrapper_name}' not found. Available: {list(WRAPPER_REGISTRY.keys())}")
    return WRAPPER_REGISTRY[wrapper_name]

