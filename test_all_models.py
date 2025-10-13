#!/usr/bin/env python3
"""
Comprehensive test module for all local model checkpoints.
Tests inference and reports parameter counts for each component.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODELS, LOCAL_MODEL_DIR
from src.model_manager import ModelManager
from src.inference import generate_image
from src.utils import get_device, get_gpu_info


def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    if model is None:
        return 0
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except:
        return sum(p.numel() for p in model.parameters())


def format_params(count: int) -> str:
    """Format parameter count in human-readable format."""
    if count == 0:
        return "0"
    elif count < 1_000:
        return f"{count}"
    elif count < 1_000_000:
        return f"{count / 1_000:.2f}K"
    elif count < 1_000_000_000:
        return f"{count / 1_000_000:.2f}M"
    else:
        return f"{count / 1_000_000_000:.2f}B"


def get_component_params(pipeline) -> Dict[str, Tuple[int, str]]:
    """Extract parameter counts for all components in a pipeline."""
    components = {}
    
    # Common component names across different pipelines
    component_attrs = [
        'text_encoder', 'text_encoder_2', 'text_encoder_3',
        'unet', 'transformer', 'vae', 'vqgan',
        'prior', 'decoder', 'prior_prior',
        'image_encoder', 'safety_checker',
        'feature_extractor', 'tokenizer',
    ]
    
    for attr in component_attrs:
        if hasattr(pipeline, attr):
            component = getattr(pipeline, attr)
            if component is not None and hasattr(component, 'parameters'):
                param_count = count_parameters(component)
                components[attr] = (param_count, format_params(param_count))
    
    return components


def check_model_exists_locally(model_id: str) -> bool:
    """Check if a model exists in the local directory."""
    local_path = os.path.join(LOCAL_MODEL_DIR, model_id)
    return os.path.exists(local_path)


def test_model_inference(
    model_id: str,
    manager: ModelManager,
    test_prompt: str = "A beautiful sunset over mountains",
    seed: int = 42
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Test inference for a single model.
    
    Returns:
        (success, error_message, result_info)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}")
    
    try:
        # Load the model
        print("Loading model...")
        pipeline = manager.load_model(model_id)
        
        # Get component parameters
        print("Analyzing components...")
        components = get_component_params(pipeline)
        
        # Calculate total parameters
        total_params = sum(count for count, _ in components.values())
        
        print("\nComponent Parameters:")
        for comp_name, (count, formatted) in sorted(components.items()):
            print(f"  - {comp_name:20s}: {formatted:>12s} ({count:,})")
        print(f"  {'Total':20s}: {format_params(total_params):>12s} ({total_params:,})")
        
        # Test inference
        print(f"\nRunning inference with prompt: '{test_prompt}'")
        image, filepath, seed_used = generate_image(
            model_id=model_id,
            prompt=test_prompt,
            num_inference_steps=20,  # Faster for testing
            guidance_scale=7.5,
            width=512,
            height=512,
            seed=seed,
            scheduler=None,
        )
        
        if image is None:
            return False, f"Inference failed: {filepath}", None
        
        print(f"✓ Inference successful!")
        print(f"  Image saved to: {filepath}")
        print(f"  Seed used: {seed_used}")
        
        result_info = {
            "model_id": model_id,
            "status": "success",
            "components": {k: {"count": v[0], "formatted": v[1]} for k, v in components.items()},
            "total_params": total_params,
            "total_params_formatted": format_params(total_params),
            "image_path": filepath,
            "seed": seed_used,
        }
        
        return True, None, result_info
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"✗ Test failed: {str(e)}")
        print(traceback.format_exc())
        return False, error_msg, None


def test_all_models(
    test_prompt: str = "A beautiful sunset over mountains",
    seed: int = 42,
    skip_missing: bool = True,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test all local model checkpoints.
    
    Returns:
        Dictionary with test results and parameter counts
    """
    print("="*80)
    print("COMPREHENSIVE MODEL TESTING")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Device: {get_device()}")
    print(f"  GPU Info:\n{get_gpu_info()}")
    print(f"  Local Model Directory: {LOCAL_MODEL_DIR}")
    print(f"  Test Prompt: {test_prompt}")
    print(f"  Seed: {seed}")
    print(f"  Skip Missing Models: {skip_missing}")
    
    # Initialize model manager
    manager = ModelManager()
    
    # Filter local models only (exclude API models)
    if model_id:
        local_models = [model_id]
        print(f"\nTesting specific model: {model_id}")
    else:
        local_models = list(MODELS.keys())
        print(f"\nTotal models to test: {len(local_models)}")
    
    # Check which models exist locally
    available_models = []
    missing_models = []
    
    for model_id in local_models:
        if check_model_exists_locally(model_id):
            available_models.append(model_id)
        else:
            missing_models.append(model_id)
    
    print(f"\nModels available locally: {len(available_models)}")
    for model_id in available_models:
        print(f"  ✓ {model_id}")
    
    if missing_models:
        print(f"\nModels NOT found locally: {len(missing_models)}")
        for model_id in missing_models:
            print(f"  ✗ {model_id}")
    
    # Test models
    models_to_test = available_models if skip_missing else local_models
    print(f"\nTesting {len(models_to_test)} models...\n")
    
    results = {
        "test_config": {
            "timestamp": datetime.now().isoformat(),
            "device": get_device(),
            "gpu_info": get_gpu_info(),
            "test_prompt": test_prompt,
            "seed": seed,
            "total_models": len(local_models),
            "available_models": len(available_models),
            "missing_models": len(missing_models),
        },
        "models": {},
        "summary": {
            "total_tested": 0,
            "successful": 0,
            "failed": 0,
        }
    }
    
    for idx, model_id in enumerate(models_to_test, 1):
        print(f"\n[{idx}/{len(models_to_test)}] Testing {model_id}...")
        
        success, error, info = test_model_inference(
            model_id, manager, test_prompt, seed
        )
        
        results["summary"]["total_tested"] += 1
        
        if success:
            results["summary"]["successful"] += 1
            results["models"][model_id] = info
        else:
            results["summary"]["failed"] += 1
            results["models"][model_id] = {
                "model_id": model_id,
                "status": "failed",
                "error": error,
            }
        
        # Unload model to free memory for next test
        manager.unload_model(model_id)
        
        # Save progressive results after each test
        try:
            output_path = Path("test_results_progress.json")
            output_path.write_text(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Warning: Could not save progressive results: {e}")
    
    return results


def generate_report(results: Dict[str, Any], output_file: str = "test_results.json"):
    """Generate a comprehensive test report."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    summary = results["summary"]
    print(f"\nTotal Models Tested: {summary['total_tested']}")
    print(f"  ✓ Successful: {summary['successful']}")
    print(f"  ✗ Failed: {summary['failed']}")
    
    # Successful models
    successful_models = [
        (model_id, info) for model_id, info in results["models"].items()
        if info["status"] == "success"
    ]
    
    if successful_models:
        print("\n" + "="*80)
        print("SUCCESSFUL MODELS - PARAMETER COUNTS")
        print("="*80)
        
        for model_id, info in successful_models:
            print(f"\n{model_id}")
            print(f"  Total Parameters: {info['total_params_formatted']} ({info['total_params']:,})")
            print(f"  Components:")
            for comp_name, comp_info in sorted(info["components"].items()):
                print(f"    - {comp_name:20s}: {comp_info['formatted']:>12s} ({comp_info['count']:,})")
    
    # Failed models
    failed_models = [
        (model_id, info) for model_id, info in results["models"].items()
        if info["status"] == "failed"
    ]
    
    if failed_models:
        print("\n" + "="*80)
        print("FAILED MODELS")
        print("="*80)
        
        for model_id, info in failed_models:
            print(f"\n{model_id}")
            print(f"  Error: {info.get('error', 'Unknown error')[:200]}...")
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*80}")
    print(f"Full results saved to: {output_path.absolute()}")
    print("="*80)


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test all local model checkpoints and report parameter counts"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over mountains",
        help="Test prompt for inference"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="Output file for test results"
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Attempt to test models not found locally (will download)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Test a specific model ID instead of all"
    )
    
    args = parser.parse_args()
    
    try:
        # Run tests
        results = test_all_models(
            test_prompt=args.prompt,
            seed=args.seed,
            skip_missing=not args.include_missing,
            model_id=args.model_id,
        )
        
        # Generate report
        generate_report(results, args.output)
        
        print("\n✓ Testing complete!")
        
    except KeyboardInterrupt:
        print("\n\n✗ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

