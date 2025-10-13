"""
Custom Gradio widget for the SeedDream image generation API.
"""

import gradio as gr
from typing import Tuple, Optional
from PIL import Image
import traceback

from src.api_clients import get_api_client
from src.utils import save_image, seed_everything
from src.config import CLOSED_SOURCE_MODELS


def generate_with_api(
    prompt: str, width: int = 1024, height: int = 1024, seed: int = -1,
) -> Tuple[Optional[Image.Image], str, int]:
    """Generate image using the SeedDream API."""
    seed_used = seed_everything(seed)
    provider = "seeddream"
    
    try:
        client = get_api_client(provider)
        if not client:
            return None, f"Unknown provider: {provider}", seed_used
        
        config = CLOSED_SOURCE_MODELS.get(provider, {})
        model = config.get("default_model")
        
        gen_kwargs = {"prompt": prompt, "width": width, "height": height, "seed": seed_used}
        
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
    """Create the SeedDream image generation widget."""
    with gr.Blocks() as widget:
        gr.Markdown(
            """
            ## 🌱 SeedDream Image Generation
            Generate images using the SeedDream 3.0 API.
            **Note:** Requires `SEEDDREAM_ACCESS_KEY_ID` and `SEEDDREAM_SECRET_ACCESS_KEY` to be set as environment variables.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                api_prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt here...",
                    lines=3,
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    api_width = gr.Slider(
                        minimum=512, maximum=2048, value=1328, step=64, label="Width",
                    )
                    api_height = gr.Slider(
                        minimum=512, maximum=2048, value=1328, step=64, label="Height",
                    )
                    api_seed = gr.Number(
                        label="Seed", value=-1, precision=0, info="-1 for random",
                    )
                
                api_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                
                api_status = gr.Textbox(label="Status", interactive=False, lines=2)
            
            with gr.Column(scale=1):
                gr.Markdown("### Generated Image")
                api_output = gr.Image(label="Result", type="pil", show_label=False)
                
                gr.Markdown(
                    """
                    ---
                    **API Key Setup:**
                    
                    Set the following environment variables:
                    - `SEEDDREAM_ACCESS_KEY_ID`
                    - `SEEDDREAM_SECRET_ACCESS_KEY`
                    
                    Images are saved to `outputs/seeddream/` directory.
                    """
                )

        def generate_handler(prompt, width, height, seed, progress=gr.Progress()):
            """Handle image generation request."""
            if not prompt:
                return None, "⚠️ Please enter a prompt!"
            
            progress(0, desc="Generating with SeedDream...")
            
            image, result, seed_used = generate_with_api(
                prompt=prompt,
                width=int(width),
                height=int(height),
                seed=int(seed),
            )
            
            if image:
                status = f"✅ Generated successfully! (seed: {seed_used})\nSaved to: {result}"
            else:
                status = f"❌ Generation failed:\n{result}"
            
            return image, status
        
        api_generate_btn.click(
            fn=generate_handler,
            inputs=[api_prompt_input, api_width, api_height, api_seed],
            outputs=[api_output, api_status],
        )
    
    return widget
