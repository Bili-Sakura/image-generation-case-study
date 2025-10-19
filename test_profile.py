"""
Profiling script for diffusion models using calflops.
Uses ModelManager for proper loading/unloading and profiler.summarize() for profiling.
Automatically captures real inputs using direct pipeline method calls.
"""

import sys
sys.path.append("/data/liuzicheng/zhenyuan/projects/image-generation-case-study")

import os
import json
from pathlib import Path

from src.profiler import create_profiler
from src.config import FLOW_MATCHING_MODELS, DIFFUSION_MODELS, SPECIAL_SCHEDULER_MODELS, MODELS
from src.model_manager import ModelManager
from src.utils import get_device

# Configuration
HEIGHT = 512
WIDTH = 512
STEPS = 50
PROMPT = "a beautiful landscape"
GUIDANCE_SCALE = 7.5

# Output directory for profiling results
OUTPUT_DIR = Path("outputs/profiling_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create profiler
profiler = create_profiler(enabled=True)
print("✓ Profiler created")

# Create model manager (no device_map for profiling to keep it simple)
model_manager = ModelManager(use_device_map=False)
print(f"✓ Model manager created (device: {model_manager.device})")

# Combine all model IDs to profile
all_model_ids = FLOW_MATCHING_MODELS + DIFFUSION_MODELS + SPECIAL_SCHEDULER_MODELS
print(f"\n📊 Profiling {len(all_model_ids)} models...\n")

# Store all results
all_results = []

for i, model_id in enumerate(all_model_ids, 1):
    model_name = MODELS.get(model_id, {}).get('short_name', model_id)
    print(f"\n{'='*80}")
    print(f"[{i}/{len(all_model_ids)}] Profiling: {model_name} ({model_id})")
    print(f"{'='*80}")
    
    try:
        # Load model
        print(f"🔄 Loading model...")
        pipe = model_manager.load_model(model_id, force_reload=False)
        
        # Profile complete generation
        print(f"📊 Running profiler...")
        summary = profiler.summarize(
            pipe=pipe,
            model_id=model_id,
            height=HEIGHT,
            width=WIDTH,
            steps=STEPS,
            prompt=PROMPT,
            guidance_scale=GUIDANCE_SCALE,
            verbose=True
        )
        
        # Add model metadata to summary
        summary["model_id"] = model_id
        summary["model_name"] = model_name
        
        # Print summary
        if summary.get("enabled", False):
            print(f"\n✅ Results for {model_name}:")
            print(f"   Architecture: {summary.get('architecture', 'N/A')}")
            print(f"   Total Parameters: {summary.get('total_params_str', 'N/A')}")
            print(f"   Main Model MACs/step: {summary.get('main_model_per_step_macs_str', 'N/A')}")
            print(f"   VAE MACs: {summary.get('vae_macs_str', 'N/A')}")
            print(f"   Text Encoder MACs: {summary.get('text_encoder_total_macs_str', 'N/A')}")
            print(f"   Total MACs ({STEPS} steps): {summary.get(f'total_{STEPS}_steps_macs_str', 'N/A')}")
        else:
            print(f"\n⚠️  Profiling disabled or failed for {model_name}")
        
        all_results.append(summary)
        
        # Unload model to free memory (following app_dev.py pattern)
        print(f"🗑️  Unloading model to free memory...")
        model_manager.unload_model(model_id)
        
    except Exception as e:
        print(f"\n❌ Error profiling {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to unload even if there was an error
        try:
            model_manager.unload_model(model_id)
        except:
            pass

# Save all results to JSON
output_file = OUTPUT_DIR / f"profiling_results_{HEIGHT}x{WIDTH}_{STEPS}steps.json"
print(f"\n{'='*80}")
print(f"💾 Saving results to: {output_file}")
with open(output_file, 'w') as f:
    json.dump({
        "config": {
            "height": HEIGHT,
            "width": WIDTH,
            "steps": STEPS,
            "prompt": PROMPT,
            "guidance_scale": GUIDANCE_SCALE,
        },
        "results": all_results
    }, f, indent=2)

print(f"✅ Profiling complete! Results saved to {output_file}")




