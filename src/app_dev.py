"""
Developer mode application with pre-selected models.
This version allows developers to select specific models and load them all at once.
"""

import gradio as gr
from typing import List

from src.config import MODELS, IMAGE_SIZE_PRESETS, DEFAULT_CONFIG
from src.model_manager import ModelManager
from src.inference import generate_images_sequential
from src.utils import get_device, estimate_memory_usage, create_and_save_image_grid


# Global model manager for developer mode
dev_model_manager = None
loaded_model_ids = []


def initialize_dev_models(selected_models: List[str], progress=gr.Progress()) -> str:
    """Initialize selected models for developer mode.

    Args:
        selected_models: List of model IDs to load
        progress: Gradio progress tracker

    Returns:
        Status message
    """
    global dev_model_manager, loaded_model_ids

    if not selected_models:
        return "❌ No models selected. Please select at least one model."

    # Create new model manager
    dev_model_manager = ModelManager()
    loaded_model_ids = []

    status_lines = ["Loading models..."]

    for i, model_id in enumerate(selected_models):
        progress(
            (i, len(selected_models)),
            desc=f"Loading {MODELS[model_id]['short_name']}...",
        )
        try:
            dev_model_manager.load_model(model_id)
            loaded_model_ids.append(model_id)
            status_lines.append(f"✓ {MODELS[model_id]['short_name']}")
        except Exception as e:
            status_lines.append(f"✗ {MODELS[model_id]['short_name']}: {str(e)}")

    if loaded_model_ids:
        status_lines.append(
            f"\n🎉 Successfully loaded {len(loaded_model_ids)}/{len(selected_models)} models"
        )
        return "\n".join(status_lines)
    else:
        return "❌ Failed to load any models. Check console for details."


def generate_with_loaded_models(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: str,
    seed: int,
    progress=gr.Progress(),
) -> List:
    """Generate images using pre-loaded models.

    Args:
        prompt: Text prompt
        negative_prompt: Negative prompt
        num_inference_steps: Number of steps
        guidance_scale: Guidance scale
        image_size: Image size preset
        seed: Random seed
        progress: Progress tracker

    Returns:
        Gallery items
    """
    global dev_model_manager, loaded_model_ids

    if not dev_model_manager or not loaded_model_ids:
        gr.Warning("No models loaded! Please initialize models first.")
        return []

    if not prompt:
        gr.Warning("Please enter a prompt!")
        return []

    # Get dimensions
    width, height = IMAGE_SIZE_PRESETS.get(image_size, (1024, 1024))

    # Generate images
    results = generate_images_sequential(
        model_ids=loaded_model_ids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        progress_callback=lambda msg: progress(msg),
    )

    # Format for gallery
    gallery_items = []
    for model_id, image, filepath, seed_used in results:
        if image is not None:
            caption = f"{MODELS[model_id]['short_name']} (seed: {seed_used})"
            gallery_items.append((filepath, caption))

    return gallery_items


def create_image_grid_dev(gallery_data: List, rows: int, cols: int):
    """Create an image grid from gallery images."""
    if not gallery_data:
        return None, "❌ No images available. Please generate images first."
    if rows <= 0 or cols <= 0:
        return None, "❌ Rows and columns must be positive numbers."
    
    image_paths = [path for path, _ in gallery_data]
    grid_path = create_and_save_image_grid(image_paths, rows=int(rows), cols=int(cols))
    
    return (grid_path, f"✓ Image grid created successfully: {grid_path}") if grid_path else (None, "❌ Failed to create image grid.")


def create_dev_ui() -> gr.Blocks:
    """Create developer mode UI."""

    device = get_device()

    with gr.Blocks(title="Dev Mode: Text-to-Image", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🔧 Developer Mode: Text-to-Image Generation
            
            **Developer Mode Features:**
            - Select specific models to work with
            - Load all selected models at once (one-time loading)
            - Fast sequential inference with pre-loaded models
            """
        )

        gr.Markdown(f"**Device:** {device}")

        with gr.Tabs():
            # Tab 1: Model Setup
            with gr.Tab("1️⃣ Model Setup"):
                gr.Markdown("### Select Models to Load")
                gr.Markdown(
                    "⚠️ **Warning:** Loading multiple large models requires significant VRAM!"
                )

                model_choices = [
                    (
                        f"{info['short_name']} ({estimate_memory_usage(model_id)})",
                        model_id,
                    )
                    for model_id, info in MODELS.items()
                ]

                model_selection = gr.CheckboxGroup(
                    choices=model_choices,
                    label="Select Models",
                    info="Choose models to load into memory",
                )

                load_btn = gr.Button(
                    "🚀 Load Selected Models", variant="primary", size="lg"
                )

                load_status = gr.Textbox(
                    label="Loading Status",
                    lines=10,
                    interactive=False,
                )

                load_btn.click(
                    fn=initialize_dev_models,
                    inputs=[model_selection],
                    outputs=load_status,
                )

            # Tab 2: Generation
            with gr.Tab("2️⃣ Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Prompt")
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt...",
                            lines=4,
                        )

                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Optional negative prompt...",
                            lines=2,
                        )

                        with gr.Accordion("Settings", open=True):
                            num_steps = gr.Slider(
                                10,
                                100,
                                DEFAULT_CONFIG["num_inference_steps"],
                                step=1,
                                label="Steps",
                            )
                            guidance = gr.Slider(
                                1.0,
                                20.0,
                                DEFAULT_CONFIG["guidance_scale"],
                                step=0.5,
                                label="Guidance Scale",
                            )
                            size = gr.Dropdown(
                                list(IMAGE_SIZE_PRESETS.keys()),
                                value="512x512",
                                label="Size",
                            )
                            seed = gr.Number(value=-1, label="Seed", precision=0)

                        gen_btn = gr.Button("🎨 Generate", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gallery = gr.Gallery(
                            label="Generated Images",
                            columns=2,
                            height="auto",
                        )

                        # Image Grid Section
                        gr.Markdown("### 🔲 Create Image Grid")
                        with gr.Row():
                            grid_rows = gr.Number(
                                label="Grid Rows",
                                value=1,
                                precision=0,
                                minimum=1,
                                info="Number of rows in the grid"
                            )
                            grid_cols = gr.Number(
                                label="Grid Columns",
                                value=2,
                                precision=0,
                                minimum=1,
                                info="Number of columns in the grid"
                            )
                        
                        create_grid_btn = gr.Button(
                            "🎨 Create Image Grid", variant="secondary", size="sm"
                        )
                        
                        grid_output = gr.Image(
                            label="Image Grid",
                            type="filepath",
                            visible=True
                        )
                        
                        grid_status = gr.Textbox(
                            label="Grid Status",
                            interactive=False,
                            visible=True,
                            lines=1
                        )

                def progress_wrapper(p, np, ns, g, sz, s):
                    """Wrapper to handle progress callback."""
                    return generate_with_loaded_models(p, np, ns, g, sz, s)

                gen_btn.click(
                    fn=progress_wrapper,
                    inputs=[prompt, negative_prompt, num_steps, guidance, size, seed],
                    outputs=gallery,
                )

                create_grid_btn.click(
                    fn=create_image_grid_dev,
                    inputs=[gallery, grid_rows, grid_cols],
                    outputs=[grid_output, grid_status],
                )

    return app


def main():
    """Main entry point for developer mode."""
    print("=" * 60)
    print("Developer Mode: Text-to-Image Generation")
    print("=" * 60)
    print("\nℹ️  This is developer mode with manual model selection")
    print("   Use src/app.py for the standard user interface\n")

    app = create_dev_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
