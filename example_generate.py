#!/usr/bin/env python3
"""
Example script demonstrating programmatic use of the text-to-image generation system.
"""

from src.model_manager import get_model_manager
from src.inference import generate_image, generate_images_sequential


def example_single_model():
    """Example: Generate with a single model."""
    print("=" * 60)
    print("Example 1: Single Model Generation")
    print("=" * 60)

    # Initialize model manager
    manager = get_model_manager()

    # Load a model
    model_id = "stabilityai/stable-diffusion-2-1-base"
    print(f"\nLoading {model_id}...")
    manager.load_model(model_id)

    # Generate image
    prompt = "A cat"
    print(f"\nGenerating: {prompt}")

    image, filepath, seed, profiling_data = generate_image(
        model_id=model_id,
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        width=512,
        height=512,
        seed=42,
        scheduler="DDIMScheduler",
        enable_profiling=True,  # Enable compute profiling
    )

    if image:
        print(f"\n✓ Image saved to: {filepath}")
        print(f"  Seed used: {seed}")
        
        # Display profiling data if available
        if profiling_data and profiling_data.get("enabled"):
            print(f"\n📊 Compute Statistics:")
            print(f"  Parameters: {profiling_data.get('params_str', 'N/A')}")
            print(f"  Total FLOPs: {profiling_data.get('total_flops_str', 'N/A')}")
            print(f"  Total MACs: {profiling_data.get('total_macs_str', 'N/A')}")
            print(f"  Inference Time: {profiling_data.get('inference_time_str', 'N/A')}")
    else:
        print(f"\n✗ Generation failed: {filepath}")


def example_multiple_models():
    """Example: Compare multiple models."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Model Comparison")
    print("=" * 60)

    # Select models to compare
    model_ids = [
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ]

    # Generate with all models
    prompt = "A futuristic cyberpunk cityscape at night with neon lights"
    print(f"\nPrompt: {prompt}")
    print(f"Models: {len(model_ids)}")

    results = generate_images_sequential(
        model_ids=model_ids,
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=12345,  # Same seed for fair comparison
        scheduler="EulerDiscreteScheduler",
    )

    # Display results
    print("\nResults:")
    for model_id, image, filepath, _ in results:
        if image:
            print(f"  ✓ {model_id}")
            print(f"    → {filepath}")
        else:
            print(f"  ✗ {model_id}: {filepath}")


def example_with_negative_prompt():
    """Example: Using negative prompts."""
    print("\n" + "=" * 60)
    print("Example 3: Using Negative Prompts")
    print("=" * 60)

    manager = get_model_manager()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    print(f"\nLoading {model_id}...")
    manager.load_model(model_id)

    prompt = "A portrait of a beautiful woman"
    negative_prompt = "ugly, deformed, blurry, low quality, distorted"

    print(f"\nPrompt: {prompt}")
    print(f"Negative: {negative_prompt}")

    image, filepath, seed = generate_image(
        model_id=model_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        seed=-1,  # Random seed
        scheduler="EulerDiscreteScheduler",
    )

    if image:
        print(f"\n✓ Image saved to: {filepath}")
        print(f"  Seed used: {seed}")


def example_batch_different_sizes():
    """Example: Generate images with different sizes."""
    print("\n" + "=" * 60)
    print("Example 4: Different Image Sizes")
    print("=" * 60)

    manager = get_model_manager()
    model_id = "stabilityai/stable-diffusion-2-1"
    manager.load_model(model_id)

    prompt = "A majestic mountain landscape"
    sizes = [
        (512, 512, "Square"),
        (768, 512, "Landscape"),
        (512, 768, "Portrait"),
    ]

    print(f"\nPrompt: {prompt}")
    print(f"Generating {len(sizes)} different sizes...\n")

    for width, height, name in sizes:
        print(f"  Generating {name} ({width}x{height})...")
        image, filepath, _ = generate_image(
            model_id=model_id,
            prompt=prompt,
            width=width,
            height=height,
            seed=42,
            scheduler="EulerDiscreteScheduler",
        )

        if image:
            print(f"    ✓ Saved to: {filepath}")


def main():
    """Run all examples."""
    print("\n" + "🎨" * 30)
    print("Text-to-Image Generation Examples")
    print("🎨" * 30 + "\n")

    try:
        # Run examples
        example_single_model()

        # Uncomment to run more examples:
        # example_multiple_models()
        # example_with_negative_prompt()
        # example_batch_different_sizes()

        print("\n" + "=" * 60)
        print("✓ Examples completed!")
        print("=" * 60)
        print("\nCheck the 'outputs/' directory for generated images.")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
