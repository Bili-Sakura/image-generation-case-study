"""
Main Gradio application for text-to-image generation.
"""

import gradio as gr
from typing import List, Tuple
import torch

from src.config import MODELS, DEFAULT_MODELS, DEFAULT_CONFIG, IMAGE_SIZE_PRESETS
from src.model_manager import get_model_manager, initialize_default_models
from src.inference import generate_images_sequential
from src.utils import get_device, estimate_memory_usage, get_gpu_info, get_gpu_vram_usage, create_and_save_image_grid, get_model_params_table
from src.closed_source_widget import create_closed_source_widget, create_batch_api_interface


def create_model_checkboxes() -> List[Tuple[str, str, bool]]:
    """Create checkbox options for model selection."""
    return [
        (f"{info['short_name']} - {estimate_memory_usage(model_id)}", model_id, model_id in DEFAULT_MODELS)
        for model_id, info in MODELS.items()
    ]


def generate_images_ui(
    prompt: str,
    negative_prompt: str,
    selected_models: List[str],
    num_inference_steps: int,
    guidance_scale: float,
    image_size: str,
    seed: int,
    progress=gr.Progress(),
) -> List[Tuple[str, str]]:
    """UI function for generating images.

    Args:
        prompt: Text prompt
        negative_prompt: Negative prompt
        selected_models: List of selected model IDs
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        image_size: Image size preset name
        seed: Random seed
        progress: Gradio progress tracker

    Returns:
        List of (image_path, caption) tuples for gallery
    """
    if not prompt:
        gr.Warning("Please enter a prompt!")
        return []

    if not selected_models:
        gr.Warning("Please select at least one model!")
        return []

    # Get image dimensions
    width, height = IMAGE_SIZE_PRESETS.get(image_size, (1024, 1024))

    # Progress callback
    def update_progress(msg: str, current: int = 0, total: int = 1):
        """Update progress with message."""
        if current > 0 and total > 0:
            progress((current, total), desc=msg)
        else:
            # For messages without specific progress, just show the message
            progress(0, desc=msg)

    # Generate images
    results = generate_images_sequential(
        model_ids=selected_models,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        progress_callback=update_progress,
    )

    # Format results for gallery
    gallery_items = []
    for model_id, image, filepath, seed_used in results:
        if image is not None:
            model_name = MODELS[model_id]["short_name"]
            caption = f"{model_name} (seed: {seed_used})"
            gallery_items.append((filepath, caption))
        else:
            # Error case - filepath contains error message
            model_name = MODELS.get(model_id, {}).get("short_name", model_id)
            print(f"Failed to generate with {model_name}: {filepath}")

    if not gallery_items:
        gr.Warning("Failed to generate any images. Check console for errors.")

    return gallery_items


def load_selected_models_ui(selected_models: List[str], progress=gr.Progress()) -> str:
    """Load selected models into memory."""
    if not selected_models:
        return "No models selected."

    manager = get_model_manager()
    loaded, failed = [], []

    for i, model_id in enumerate(selected_models):
        progress((i + 1, len(selected_models)), desc=f"Loading {model_id}...")
        try:
            manager.load_model(model_id)
            loaded.append(MODELS[model_id]["short_name"])
        except Exception as e:
            failed.append((MODELS[model_id]["short_name"], str(e)))

    status_lines = []
    if loaded:
        status_lines.append(f"✓ Successfully loaded: {', '.join(loaded)}")
    if failed:
        status_lines.append("✗ Failed to load:")
        status_lines.extend([f"  - {name}: {error}" for name, error in failed])
    return "\n".join(status_lines)


def create_image_grid_ui(gallery_data: List[Tuple[str, str]], rows: int, cols: int) -> Tuple[str, str]:
    """Create an image grid from gallery images."""
    if not gallery_data:
        return None, "❌ No images available. Please generate images first."
    if rows <= 0 or cols <= 0:
        return None, "❌ Rows and columns must be positive numbers."
    
    image_paths = [path for path, _ in gallery_data]
    grid_path = create_and_save_image_grid(image_paths, rows=rows, cols=cols)
    
    return (grid_path, f"✓ Image grid created successfully: {grid_path}") if grid_path else (None, "❌ Failed to create image grid.")


def create_ui() -> gr.Blocks:
    """Create the Gradio UI.

    Returns:
        Gradio Blocks interface
    """
    # Device info
    device = get_device()
    gpu_info = get_gpu_info()

    with gr.Blocks(title="Text-to-Image Generation", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🎨 Text-to-Image Generation Case Study
            
            Generate images using multiple state-of-the-art diffusion models (open-source) 
            or commercial APIs (closed-source).
            """
        )

        gr.Markdown(f"**Device:** {device}\n\n{gpu_info}")
        
        # GPU VRAM Usage Monitor
        with gr.Accordion("📊 GPU VRAM Monitor", open=False):
            vram_display = gr.Textbox(
                label="Real-time VRAM Usage",
                value=get_gpu_vram_usage(),
                interactive=False,
                lines=8,
                show_copy_button=True,
            )
            refresh_vram_btn = gr.Button("🔄 Refresh VRAM", size="sm")
            refresh_vram_btn.click(fn=get_gpu_vram_usage, outputs=vram_display)
        
        # Auto-refresh VRAM display every 5 seconds
        vram_timer = gr.Timer(value=5, active=True)
        vram_timer.tick(fn=get_gpu_vram_usage, outputs=vram_display)
        
        # Model Parameters Table
        with gr.Accordion("📋 Model Parameters & VRAM Reference", open=False):
            gr.Markdown(get_model_params_table())
        
        # Create tabs for open-source and closed-source models
        with gr.Tabs():
            # Tab 1: Open-Source Models
            with gr.Tab("🔓 Open-Source Models"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input section
                        gr.Markdown("### 📝 Prompt")
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your text prompt here...",
                            lines=3,
                        )

                        negative_prompt_input = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="Enter negative prompt to avoid certain features...",
                            lines=2,
                        )

                        # Model selection
                        gr.Markdown("### 🤖 Model Selection")
                        model_checkboxes = create_model_checkboxes()

                        model_selector = gr.CheckboxGroup(
                            choices=[(label, value) for label, value, _ in model_checkboxes],
                            value=[
                                value for _, value, is_default in model_checkboxes if is_default
                            ],
                            label="Select Models",
                            info="Choose which models to use for generation",
                        )

                        with gr.Row():
                            load_models_btn = gr.Button("🔄 Load Selected Models", size="sm")

                        load_status = gr.Textbox(
                            label="Load Status",
                            interactive=False,
                            lines=3,
                            visible=False,
                        )

                        # Generation settings (collapsible)
                        with gr.Accordion("⚙️ Generation Settings", open=False):
                            num_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=DEFAULT_CONFIG["num_inference_steps"],
                                step=1,
                                label="Number of Inference Steps",
                                info="More steps = higher quality but slower",
                            )

                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=DEFAULT_CONFIG["guidance_scale"],
                                step=0.5,
                                label="Guidance Scale",
                                info="1.0 = faster (no CFG), higher = stronger prompt adherence",
                            )

                            image_size = gr.Dropdown(
                                choices=list(IMAGE_SIZE_PRESETS.keys()),
                                value="512x512",
                                label="Image Size",
                            )

                            seed_input = gr.Number(
                                label="Seed",
                                value=DEFAULT_CONFIG["seed"],
                                precision=0,
                                info="-1 for random seed",
                            )

                        # Generate button
                        generate_btn = gr.Button(
                            "🚀 Generate Images", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        # Output section
                        gr.Markdown("### 🖼️ Generated Images")

                        output_gallery = gr.Gallery(
                            label="Results",
                            show_label=False,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                        )

                        gr.Markdown(
                            """
                            ---
                            💡 **Tips:**
                            - Images are automatically saved to `/outputs/{timestamp}/` directory
                            - Each generation creates a new timestamped folder with all images and a config JSON
                            - Use the same seed across models for fair comparison
                            - Default models are pre-loaded for faster generation
                            """
                        )

                # Event handlers for open-source tab
                generate_btn.click(
                    fn=generate_images_ui,
                    inputs=[
                        prompt_input,
                        negative_prompt_input,
                        model_selector,
                        num_steps,
                        guidance_scale,
                        image_size,
                        seed_input,
                    ],
                    outputs=output_gallery,
                )

                load_models_btn.click(
                    fn=load_selected_models_ui,
                    inputs=[model_selector],
                    outputs=load_status,
                ).then(
                    lambda: gr.update(visible=True),
                    outputs=load_status,
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

                gr.Markdown(
                    """
                    ---
                    💡 **Tips:**
                    - Images are automatically saved to `/outputs/{timestamp}/` directory
                    - Each generation creates a new timestamped folder with all images and a config JSON
                    - Use the same seed across models for fair comparison
                    - Default models are pre-loaded for faster generation
                    - Image grids are saved to `outputs/grids/` directory
                    """
                )
            
            # Tab 2: Closed-Source APIs (Single Generation)
            with gr.Tab("🔒 Closed-Source APIs"):
                create_closed_source_widget()
            
            # Tab 3: Batch API Comparison
            with gr.Tab("🔄 API Comparison"):
                create_batch_api_interface()

        create_grid_btn.click(
            fn=create_image_grid_ui,
            inputs=[output_gallery, grid_rows, grid_cols],
            outputs=[grid_output, grid_status],
        )

    return app


def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("Text-to-Image Generation Case Study")
    print("=" * 60)
    
    # Display GPU information
    print(f"\n{get_gpu_info()}")

    # Initialize default models
    print("\nInitializing default models...")
    try:
        initialize_default_models()
        print("✓ Default models initialized successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to initialize some default models: {e}")
        print("You can still load models manually from the UI")

    # Create and launch UI
    print("\nLaunching Gradio interface...")
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
