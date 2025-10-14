"""
Batch mode application for testing all models sequentially.
This version loads, inferences, and unloads models one by one without pre-loading.
"""

import gradio as gr
from typing import List, Tuple

from src.config import MODELS, IMAGE_SIZE_PRESETS, DEFAULT_CONFIG
from src.inference import generate_all_models_sequential
from src.utils import get_device, get_gpu_info, get_gpu_vram_usage, create_and_save_image_grid, get_model_params_table


def generate_batch_all_models(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: str,
    seed: int,
    progress=gr.Progress(),
) -> List[Tuple[str, str]]:
    """Generate images with all models sequentially (load → inference → unload).

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
    if not prompt:
        gr.Warning("Please enter a prompt!")
        return []

    # Get dimensions
    width, height = IMAGE_SIZE_PRESETS.get(image_size, (1024, 1024))

    # Progress callback wrapper
    def update_progress(msg: str, current: int = 0, total: int = 1):
        """Update progress with message."""
        if current > 0 and total > 0:
            progress((current, total), desc=msg)
        else:
            progress(0, desc=msg)
    
    # Generate images with all models
    results = generate_all_models_sequential(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        progress_callback=update_progress,
    )

    # Format for gallery
    gallery_items = []
    for model_id, image, filepath, seed_used in results:
        if image is not None:
            model_name = MODELS.get(model_id, {}).get('short_name', model_id)
            caption = f"{model_name} (seed: {seed_used})"
            gallery_items.append((filepath, caption))
        else:
            # Error case - filepath contains error message
            model_name = MODELS.get(model_id, {}).get('short_name', model_id)
            print(f"Failed to generate with {model_name}: {filepath}")

    if not gallery_items:
        gr.Warning("Failed to generate any images. Check console for errors.")

    return gallery_items


def create_image_grid_batch(gallery_data: List[Tuple[str, str]], rows: int, cols: int) -> Tuple[str, str]:
    """Create an image grid from gallery images."""
    if not gallery_data:
        return None, "❌ No images available. Please generate images first."
    if rows <= 0 or cols <= 0:
        return None, "❌ Rows and columns must be positive numbers."
    
    image_paths = [path for path, _ in gallery_data]
    grid_path = create_and_save_image_grid(image_paths, rows=int(rows), cols=int(cols))
    
    return (grid_path, f"✓ Image grid created successfully: {grid_path}") if grid_path else (None, "❌ Failed to create image grid.")


def get_all_models_info() -> str:
    """Get information about all available models."""
    info_lines = [f"📊 Available models for batch processing ({len(MODELS)}):"]
    info_lines.append("")
    info_lines.append("Models will be processed in the following order:")
    info_lines.append("")
    
    for i, (model_id, model_info) in enumerate(MODELS.items(), 1):
        short_name = model_info.get('short_name', model_id)
        info_lines.append(f"  {i}. {short_name}")
    
    info_lines.append("")
    info_lines.append("Each model will be:")
    info_lines.append("  1️⃣  Loaded into memory")
    info_lines.append("  2️⃣  Used for inference")
    info_lines.append("  3️⃣  Unloaded from memory")
    info_lines.append("")
    info_lines.append("⚡ This approach minimizes memory usage but takes longer.")
    
    return "\n".join(info_lines)


def create_batch_ui() -> gr.Blocks:
    """Create batch mode UI."""

    device = get_device()
    gpu_info = get_gpu_info()

    with gr.Blocks(title="Batch Mode: All Models", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🔄 Batch Mode: Test All Models
            
            **Batch Mode Features:**
            - Automatically tests **all** available models
            - No pre-loading required
            - No model selection needed
            - Each model: Load → Inference → Unload (one by one)
            - Minimizes memory usage
            - Ideal for comprehensive testing and benchmarking
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
        
        # Available Models Info
        with gr.Accordion("📝 Models to be Processed", open=True):
            models_info = gr.Textbox(
                label="Model Processing Order",
                value=get_all_models_info(),
                interactive=False,
                lines=25,
                show_copy_button=True,
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Prompt")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt here...\nThis prompt will be used for ALL models.",
                    lines=4,
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="Enter negative prompt to avoid certain features...",
                    lines=2,
                )

                gr.Markdown("### ⚙️ Generation Settings")
                
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
                    "🚀 Generate with All Models", variant="primary", size="lg"
                )
                
                gr.Markdown(
                    """
                    ⚠️ **Note:**
                    - This will process **all {} models** sequentially
                    - Each model will be loaded, used, then unloaded
                    - Total time depends on number of models and settings
                    - Progress will be shown for each model
                    """.format(len(MODELS))
                )

            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 🖼️ Generated Images (All Models)")

                output_gallery = gr.Gallery(
                    label="Results from All Models",
                    show_label=False,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="contain",
                )
                
                # Image Grid Section
                gr.Markdown("### 🔲 Create Image Grid")
                with gr.Row():
                    grid_rows = gr.Number(
                        label="Grid Rows",
                        value=2,
                        precision=0,
                        minimum=1,
                        info="Number of rows in the grid"
                    )
                    grid_cols = gr.Number(
                        label="Grid Columns",
                        value=7,
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
                    - Use the same seed for consistent comparison across models
                    - This mode uses minimal memory since models are loaded one at a time
                    - Image grids are saved to `outputs/grids/` directory
                    """
                )

        # Event handlers
        generate_btn.click(
            fn=generate_batch_all_models,
            inputs=[
                prompt,
                negative_prompt,
                num_steps,
                guidance_scale,
                image_size,
                seed_input,
            ],
            outputs=output_gallery,
        )
        
        create_grid_btn.click(
            fn=create_image_grid_batch,
            inputs=[output_gallery, grid_rows, grid_cols],
            outputs=[grid_output, grid_status],
        )

    return app


def main():
    """Main entry point for batch mode."""
    print("=" * 60)
    print("Batch Mode: Test All Models")
    print("=" * 60)
    print(f"\nℹ️  This mode will test all {len(MODELS)} models sequentially")
    print("   Each model: Load → Inference → Unload")
    print("   No pre-loading required, minimal memory usage\n")

    app = create_batch_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()

