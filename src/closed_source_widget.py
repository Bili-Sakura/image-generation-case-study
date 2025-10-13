"""
Custom Gradio widget for closed-source image generation APIs.
"""

import gradio as gr
from typing import List, Tuple, Optional
from PIL import Image
import traceback

from src.api_clients import get_api_client
from src.utils import save_image, set_seed
from src.config import CLOSED_SOURCE_MODELS


def generate_with_api(
    provider: str, prompt: str, width: int = 1024, height: int = 1024,
    model: Optional[str] = None, quality: str = "standard", style: str = "vivid", seed: int = -1,
) -> Tuple[Optional[Image.Image], str, int]:
    """Generate image using closed-source API."""
    seed_used = set_seed(seed)
    
    try:
        client = get_api_client(provider)
        if not client:
            return None, f"Unknown provider: {provider}", seed_used
        
        config = CLOSED_SOURCE_MODELS.get(provider, {})
        model = model or config.get("default_model")
        
        gen_kwargs = {"prompt": prompt, "width": width, "height": height}
        
        # Add provider-specific parameters
        if provider == "openai":
            gen_kwargs.update({"model": model, "quality": quality, "style": style})
        elif model:
            gen_kwargs["model"] = model
        
        image, error = client.generate(**gen_kwargs)
        
        if error:
            return None, error, seed_used
        if image:
            filepath = save_image(image, f"{provider}/{model or 'default'}", seed_used, prompt)
            return image, filepath, seed_used
        return None, "Failed to generate image", seed_used
    except Exception as e:
        error_msg = f"Error generating with {provider}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg, seed_used


def create_closed_source_widget() -> gr.Blocks:
    """Create the closed-source image generation widget.
    
    Returns:
        Gradio Blocks component containing the widget
    """
    with gr.Blocks() as widget:
        gr.Markdown(
            """
            ## 🔒 Closed-Source Image Generation
            Generate images using commercial APIs from OpenAI, Google, Bytedance, and Kling.
            **Note:** Requires API keys to be set as environment variables.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Provider selection
                provider_selector = gr.Radio(
                    choices=[
                        ("OpenAI DALL-E", "openai"),
                        ("Google Imagen", "google"),
                        ("Bytedance Cloud", "bytedance"),
                        ("Kling AI", "kling"),
                    ],
                    value="openai",
                    label="Select Provider",
                    info="Choose which API to use",
                )
                
                # Prompt input
                api_prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt here...",
                    lines=3,
                )
                
                # Model selection (dynamic based on provider)
                model_selector = gr.Dropdown(
                    choices=CLOSED_SOURCE_MODELS["openai"]["models"],
                    value=CLOSED_SOURCE_MODELS["openai"]["default_model"],
                    label="Model Version",
                    info="Select specific model version",
                )
                
                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    # Image size
                    api_width = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Width",
                    )
                    
                    api_height = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Height",
                    )
                    
                    # OpenAI-specific settings
                    quality_selector = gr.Radio(
                        choices=["standard", "hd"],
                        value="standard",
                        label="Quality (DALL-E only)",
                        info="HD costs more",
                        visible=True,
                    )
                    
                    style_selector = gr.Radio(
                        choices=["vivid", "natural"],
                        value="vivid",
                        label="Style (DALL-E only)",
                        info="Vivid for hyper-real, natural for realistic",
                        visible=True,
                    )
                    
                    api_seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        info="-1 for random (note: most APIs ignore seed)",
                    )
                
                # Generate button
                api_generate_btn = gr.Button(
                    "🚀 Generate with API", variant="primary", size="lg"
                )
                
                # API status
                api_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )
            
            with gr.Column(scale=1):
                # Output
                gr.Markdown("### Generated Image")
                api_output = gr.Image(
                    label="Result",
                    type="pil",
                    show_label=False,
                )
                
                gr.Markdown(
                    """
                    ---
                    **API Key Setup:**
                    
                    Set the following environment variables:
                    - `OPENAI_API_KEY` - For OpenAI DALL-E
                    - `GOOGLE_API_KEY` - For Google Imagen
                    - `GOOGLE_PROJECT_ID` - Google Cloud Project ID
                    - `BYTEDANCE_API_KEY` - For Bytedance Cloud
                    - `KLING_API_KEY` - For Kling AI
                    
                    Images are saved to `outputs/{provider}/` directory.
                    """
                )
        
        # Dynamic UI updates based on provider selection
        def update_model_choices(provider):
            """Update model dropdown based on selected provider."""
            config = CLOSED_SOURCE_MODELS.get(provider, {})
            models = config.get("models", [])
            default = config.get("default_model", models[0] if models else None)
            
            # Show/hide provider-specific options
            show_openai_opts = provider == "openai"
            
            return (
                gr.update(choices=models, value=default),
                gr.update(visible=show_openai_opts),
                gr.update(visible=show_openai_opts),
            )
        
        provider_selector.change(
            fn=update_model_choices,
            inputs=[provider_selector],
            outputs=[model_selector, quality_selector, style_selector],
        )
        
        # Generate image handler
        def generate_handler(
            provider, prompt, width, height, model, quality, style, seed, progress=gr.Progress()
        ):
            """Handle image generation request."""
            if not prompt:
                return None, "⚠️ Please enter a prompt!"
            
            progress(0, desc=f"Generating with {CLOSED_SOURCE_MODELS[provider]['short_name']}...")
            
            image, result, seed_used = generate_with_api(
                provider=provider,
                prompt=prompt,
                width=int(width),
                height=int(height),
                model=model,
                quality=quality,
                style=style,
                seed=int(seed),
            )
            
            if image:
                status = f"✅ Generated successfully! (seed: {seed_used})\nSaved to: {result}"
            else:
                status = f"❌ Generation failed:\n{result}"
            
            return image, status
        
        api_generate_btn.click(
            fn=generate_handler,
            inputs=[
                provider_selector,
                api_prompt_input,
                api_width,
                api_height,
                model_selector,
                quality_selector,
                style_selector,
                api_seed,
            ],
            outputs=[api_output, api_status],
        )
    
    return widget


def create_batch_api_interface() -> gr.Blocks:
    """Create interface for batch generation with multiple APIs.
    
    Returns:
        Gradio Blocks component for batch API generation
    """
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            ## 🔄 Batch API Generation
            Generate images using multiple closed-source APIs simultaneously for comparison.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                batch_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt...",
                    lines=3,
                )
                
                batch_providers = gr.CheckboxGroup(
                    choices=[
                        ("OpenAI DALL-E", "openai"),
                        ("Google Imagen", "google"),
                        ("Bytedance Cloud", "bytedance"),
                        ("Kling AI", "kling"),
                    ],
                    value=["openai"],
                    label="Select Providers",
                    info="Generate with multiple APIs",
                )
                
                with gr.Row():
                    batch_width = gr.Number(label="Width", value=1024, precision=0)
                    batch_height = gr.Number(label="Height", value=1024, precision=0)
                
                batch_generate_btn = gr.Button("🚀 Generate All", variant="primary")
            
            with gr.Column(scale=2):
                batch_gallery = gr.Gallery(
                    label="Results",
                    show_label=True,
                    columns=2,
                    rows=2,
                    height="auto",
                )
                
                batch_status = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                )
        
        def batch_generate_handler(prompt, providers, width, height, progress=gr.Progress()):
            """Generate with multiple providers."""
            if not prompt:
                return [], "⚠️ Please enter a prompt!"
            
            if not providers:
                return [], "⚠️ Please select at least one provider!"
            
            results = []
            status_lines = []
            
            for i, provider in enumerate(providers):
                progress((i, len(providers)), desc=f"Generating with {provider}...")
                
                image, result, seed = generate_with_api(
                    provider=provider,
                    prompt=prompt,
                    width=int(width),
                    height=int(height),
                )
                
                provider_name = CLOSED_SOURCE_MODELS[provider]["short_name"]
                
                if image:
                    results.append((result, f"{provider_name} (seed: {seed})"))
                    status_lines.append(f"✅ {provider_name}: Success")
                else:
                    status_lines.append(f"❌ {provider_name}: {result}")
            
            return results, "\n".join(status_lines)
        
        batch_generate_btn.click(
            fn=batch_generate_handler,
            inputs=[batch_prompt, batch_providers, batch_width, batch_height],
            outputs=[batch_gallery, batch_status],
        )
    
    return interface
