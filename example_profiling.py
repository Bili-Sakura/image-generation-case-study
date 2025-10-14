#!/usr/bin/env python3
"""
Example script demonstrating compute profiling (FLOPs/MACs calculation) during inference.
This shows how to measure the computational cost of different models.
"""

from src.model_manager import get_model_manager
from src.inference import generate_image
import json


def profile_single_model():
    """Profile a single model and display compute statistics."""
    print("=" * 70)
    print("Example: Profiling a Single Model")
    print("=" * 70)

    manager = get_model_manager()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    print(f"\nLoading model: {model_id}")
    manager.load_model(model_id)

    prompt = "A beautiful landscape with mountains and a lake"
    print(f"\nGenerating image with profiling enabled...")
    print(f"Prompt: {prompt}\n")

    image, filepath, seed, profiling_data = generate_image(
        model_id=model_id,
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        width=512,
        height=512,
        seed=42,
        enable_profiling=True,  # Enable compute profiling
    )

    if image and profiling_data and profiling_data.get("enabled"):
        print("\n" + "=" * 70)
        print("📊 COMPUTE PROFILING RESULTS")
        print("=" * 70)
        
        print(f"\nModel: {model_id}")
        print(f"Component: {profiling_data.get('model_component', 'N/A')}")
        print(f"\nModel Size:")
        print(f"  Parameters: {profiling_data.get('params_str', 'N/A')} ({profiling_data.get('total_params', 0):,} params)")
        
        print(f"\nCompute Cost (per step):")
        print(f"  FLOPs: {profiling_data.get('flops_per_step_str', 'N/A')}")
        print(f"  MACs:  {profiling_data.get('macs_per_step_str', 'N/A')}")
        
        print(f"\nTotal Compute Cost ({profiling_data.get('num_inference_steps', 0)} steps):")
        print(f"  FLOPs: {profiling_data.get('total_flops_str', 'N/A')} ({profiling_data.get('total_flops', 0):,} FLOPs)")
        print(f"  MACs:  {profiling_data.get('total_macs_str', 'N/A')} ({profiling_data.get('total_macs', 0):,} MACs)")
        
        print(f"\nInference Performance:")
        print(f"  Time: {profiling_data.get('inference_time_str', 'N/A')}")
        
        # Calculate efficiency metrics
        if profiling_data.get('total_flops', 0) > 0 and profiling_data.get('inference_time_seconds', 0) > 0:
            flops_per_second = profiling_data['total_flops'] / profiling_data['inference_time_seconds']
            print(f"  Throughput: {flops_per_second/1e12:.2f} TFLOP/s")
        
        print(f"\nImage saved to: {filepath}")
        print("=" * 70)
        
        # Save profiling data to JSON
        output_json = filepath.replace('.png', '_profile.json')
        with open(output_json, 'w') as f:
            json.dump(profiling_data, f, indent=2)
        print(f"\nProfiling data saved to: {output_json}")
        
    elif not profiling_data or not profiling_data.get("enabled"):
        print("\n⚠️  Profiling was disabled or failed.")
        print("  Make sure 'calflops' is installed: pip install calflops")
    else:
        print(f"\n✗ Generation failed")


def compare_models_compute_cost():
    """Compare compute cost across different models."""
    print("\n" + "=" * 70)
    print("Example: Comparing Compute Cost Across Models")
    print("=" * 70)

    manager = get_model_manager()
    
    # Select models to compare
    model_ids = [
        "stabilityai/stable-diffusion-2-1-base",
        "PixArt-alpha/PixArt-XL-2-512x512",
    ]
    
    prompt = "A futuristic city at sunset"
    results = []
    
    print(f"\nPrompt: {prompt}")
    print(f"Models to compare: {len(model_ids)}\n")
    
    for model_id in model_ids:
        print(f"\n{'='*70}")
        print(f"Processing: {model_id}")
        print(f"{'='*70}")
        
        try:
            manager.load_model(model_id)
            
            image, filepath, seed, profiling_data = generate_image(
                model_id=model_id,
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                width=512,
                height=512,
                seed=42,  # Same seed for fair comparison
                enable_profiling=True,
            )
            
            if profiling_data and profiling_data.get("enabled"):
                results.append({
                    "model_id": model_id,
                    "params": profiling_data.get("total_params", 0),
                    "params_str": profiling_data.get("params_str", "N/A"),
                    "total_flops": profiling_data.get("total_flops", 0),
                    "total_flops_str": profiling_data.get("total_flops_str", "N/A"),
                    "total_macs": profiling_data.get("total_macs", 0),
                    "total_macs_str": profiling_data.get("total_macs_str", "N/A"),
                    "inference_time": profiling_data.get("inference_time_seconds", 0),
                    "inference_time_str": profiling_data.get("inference_time_str", "N/A"),
                    "filepath": filepath,
                })
            
            # Unload to free memory
            manager.unload_model(model_id)
            
        except Exception as e:
            print(f"✗ Error with {model_id}: {e}")
    
    # Display comparison table
    if results:
        print("\n" + "=" * 70)
        print("📊 COMPUTE COST COMPARISON")
        print("=" * 70)
        
        print(f"\n{'Model':<40} {'Params':<12} {'FLOPs':<15} {'MACs':<15} {'Time':<10}")
        print("-" * 92)
        
        for result in results:
            model_name = result['model_id'].split('/')[-1][:38]
            print(f"{model_name:<40} {result['params_str']:<12} {result['total_flops_str']:<15} {result['total_macs_str']:<15} {result['inference_time_str']:<10}")
        
        print("=" * 70)
        
        # Find most efficient model (lowest FLOPs)
        if all(r['total_flops'] > 0 for r in results):
            most_efficient = min(results, key=lambda x: x['total_flops'])
            print(f"\n✨ Most compute-efficient: {most_efficient['model_id']}")
            print(f"   ({most_efficient['total_flops_str']} FLOPs)")


def profile_with_different_resolutions():
    """Profile the same model with different image resolutions."""
    print("\n" + "=" * 70)
    print("Example: Profiling Different Resolutions")
    print("=" * 70)

    manager = get_model_manager()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    print(f"\nLoading model: {model_id}")
    manager.load_model(model_id)

    prompt = "A serene garden"
    resolutions = [
        (512, 512, "512x512"),
        (768, 768, "768x768"),
        (1024, 1024, "1024x1024"),
    ]
    
    results = []
    
    print(f"\nPrompt: {prompt}")
    print(f"Testing {len(resolutions)} different resolutions...\n")
    
    for width, height, name in resolutions:
        print(f"\n{'='*70}")
        print(f"Resolution: {name} ({width}x{height})")
        print(f"{'='*70}")
        
        image, filepath, seed, profiling_data = generate_image(
            model_id=model_id,
            prompt=prompt,
            num_inference_steps=50,
            width=width,
            height=height,
            seed=42,
            enable_profiling=True,
        )
        
        if profiling_data and profiling_data.get("enabled"):
            results.append({
                "resolution": name,
                "width": width,
                "height": height,
                "total_flops_str": profiling_data.get("total_flops_str", "N/A"),
                "total_macs_str": profiling_data.get("total_macs_str", "N/A"),
                "inference_time_str": profiling_data.get("inference_time_str", "N/A"),
            })
    
    # Display results
    if results:
        print("\n" + "=" * 70)
        print("📊 RESOLUTION IMPACT ON COMPUTE COST")
        print("=" * 70)
        
        print(f"\n{'Resolution':<15} {'FLOPs':<20} {'MACs':<20} {'Time':<10}")
        print("-" * 65)
        
        for result in results:
            print(f"{result['resolution']:<15} {result['total_flops_str']:<20} {result['total_macs_str']:<20} {result['inference_time_str']:<10}")
        
        print("=" * 70)


def main():
    """Run profiling examples."""
    print("\n" + "🔬" * 35)
    print("COMPUTE PROFILING EXAMPLES - FLOPs & MACs Measurement")
    print("🔬" * 35 + "\n")

    try:
        # Example 1: Profile a single model
        # profile_single_model()
        
        # Example 2: Compare models (uncomment to run)
        compare_models_compute_cost()
        
        # Example 3: Different resolutions (uncomment to run)
        # profile_with_different_resolutions()
        
        print("\n" + "=" * 70)
        print("✓ Profiling examples completed!")
        print("=" * 70)
        print("\nNote: Profiling data is also saved in the generation config JSON")
        print("      files in the outputs/ directory.")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

