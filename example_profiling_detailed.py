#!/usr/bin/env python3
"""
Example script demonstrating detailed compute profiling using thop library.
This shows MACs breakdown for UNet, VAE, and Text Encoder separately.
"""

from src.model_manager import get_model_manager
from src.compute_profiler import create_profiler


def profile_with_breakdown():
    """Profile a model and show detailed component-wise breakdown."""
    print("=" * 70)
    print("Example: Detailed Component-wise Profiling")
    print("=" * 70)

    manager = get_model_manager()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    print(f"\nLoading model: {model_id}")
    pipe = manager.load_model(model_id)

    # Create profiler
    profiler = create_profiler(enabled=True)
    
    # Profile with detailed breakdown
    height, width = 512, 512
    steps = 30
    prompt = "a photo of a cat"
    guidance_scale = 7.5
    
    print(f"\nProfiling configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Steps: {steps}")
    print(f"  Prompt: {prompt}")
    print(f"  Guidance Scale: {guidance_scale}")
    
    print(f"\n{'=' * 70}")
    print("📊 COMPUTING MACs BREAKDOWN")
    print(f"{'=' * 70}\n")
    
    # Get detailed summary
    summary = profiler.summarize_macs(
        pipe=pipe,
        height=height,
        width=width,
        steps=steps,
        prompt=prompt,
        guidance_scale=guidance_scale,
    )
    
    if summary.get("enabled") == False:
        print("⚠️  Profiling failed. Make sure thop is installed.")
        print("   Install with: pip install thop")
        return
    
    # Display results
    print("Component-wise MACs Breakdown:")
    print("-" * 70)
    print(f"  UNet (per step):       {summary['UNet per-step (GMACs)']:>10.3f} GMACs")
    print(f"  Text Encoder (once):   {summary['Text encoder once (GMACs)']:>10.3f} GMACs")
    print(f"  VAE Decoder (once):    {summary['VAE decode once (GMACs)']:>10.3f} GMACs")
    print("-" * 70)
    print(f"  Total ({steps} steps):     {summary[f'Total {steps} steps (GMACs)']:>10.3f} GMACs")
    print("=" * 70)
    
    # Calculate percentages
    total = summary[f'Total {steps} steps (GMACs)']
    unet_total = summary['UNet per-step (GMACs)'] * steps
    text_total = summary['Text encoder once (GMACs)']
    vae_total = summary['VAE decode once (GMACs)']
    
    print("\nPercentage Breakdown:")
    print("-" * 70)
    if total > 0:
        print(f"  UNet:          {(unet_total/total)*100:>6.2f}%")
        print(f"  Text Encoder:  {(text_total/total)*100:>6.2f}%")
        print(f"  VAE Decoder:   {(vae_total/total)*100:>6.2f}%")
    print("=" * 70)
    
    # Show scaling with steps
    print("\n📈 Compute Scaling with Steps:")
    print("-" * 70)
    for step_count in [10, 20, 30, 50, 100]:
        total_for_steps = (
            summary['UNet per-step (GMACs)'] * step_count +
            summary['Text encoder once (GMACs)'] +
            summary['VAE decode once (GMACs)']
        )
        print(f"  {step_count:>3} steps: {total_for_steps:>10.3f} GMACs")
    print("=" * 70)


def compare_resolutions():
    """Compare compute cost across different resolutions."""
    print("\n" + "=" * 70)
    print("Example: Resolution Impact on Compute")
    print("=" * 70)

    manager = get_model_manager()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    print(f"\nLoading model: {model_id}")
    pipe = manager.load_model(model_id)

    profiler = create_profiler(enabled=True)
    
    resolutions = [
        (256, 256),
        (512, 512),
        (768, 768),
        (1024, 1024),
    ]
    
    steps = 30
    prompt = "a photo"
    guidance_scale = 7.5
    
    print(f"\nTesting {len(resolutions)} resolutions with {steps} steps...")
    print(f"\n{'Resolution':<15} {'Total GMACs':<15} {'UNet/step':<15} {'Relative Cost':<15}")
    print("-" * 70)
    
    results = []
    for width, height in resolutions:
        summary = profiler.summarize_macs(
            pipe=pipe,
            height=height,
            width=width,
            steps=steps,
            prompt=prompt,
            guidance_scale=guidance_scale,
        )
        
        if summary.get("enabled") != False:
            total = summary[f'Total {steps} steps (GMACs)']
            unet_per_step = summary['UNet per-step (GMACs)']
            results.append((width, height, total, unet_per_step))
    
    # Display results
    baseline_cost = results[0][2] if results else 1.0
    for width, height, total, unet_per_step in results:
        relative = total / baseline_cost
        print(f"{width}x{height:<10} {total:<15.3f} {unet_per_step:<15.3f} {relative:<15.2f}x")
    
    print("=" * 70)
    print("\n💡 Note: Compute cost scales quadratically with resolution")
    print("   (due to spatial attention mechanisms)")


def main():
    """Run detailed profiling examples."""
    print("\n" + "🔬" * 35)
    print("DETAILED COMPUTE PROFILING - Component-wise MACs Analysis")
    print("🔬" * 35 + "\n")

    try:
        # Example 1: Detailed breakdown
        profile_with_breakdown()
        
        # Example 2: Resolution comparison
        compare_resolutions()
        
        print("\n" + "=" * 70)
        print("✓ Detailed profiling examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
