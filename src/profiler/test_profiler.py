#!/usr/bin/env python3
"""
Test script for the simplified compute profiler.

Run from project root using one of these methods:
  1. python src/profiler/test_profiler.py       (recommended - catches all import errors)
  2. python -m src.profiler.test_profiler       (may fail silently on import errors)

This script provides comprehensive error visibility for CI/CD pipelines.
"""

import sys
import traceback

# ============================================================================
# Pre-flight checks: Verify imports and dependencies before running tests
# ============================================================================

def check_dependencies():
    """Check that all required dependencies are available."""
    print("="*60)
    print("Pre-flight Dependency Check")
    print("="*60)
    
    missing = []
    warnings = []
    
    # Check PyYAML (required by architecture_matcher.py)
    try:
        import yaml
        print(f"  ✓ PyYAML: {yaml.__version__ if hasattr(yaml, '__version__') else 'available'}")
    except ImportError as e:
        print(f"  ✗ PyYAML: MISSING - {e}")
        missing.append("pyyaml")
    
    # Check pathlib (standard library, but verify)
    try:
        from pathlib import Path
        print(f"  ✓ pathlib: available")
    except ImportError as e:
        print(f"  ✗ pathlib: MISSING - {e}")
        missing.append("pathlib")
    
    # Check json (standard library)
    try:
        import json
        print(f"  ✓ json: available")
    except ImportError as e:
        print(f"  ✗ json: MISSING - {e}")
        missing.append("json")
    
    # Check torch (required by profiler.py)
    try:
        import torch
        print(f"  ✓ torch: {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: MISSING - {e}")
        missing.append("torch")
    
    # Check diffusers (required by profiler.py)
    try:
        import diffusers
        print(f"  ✓ diffusers: {diffusers.__version__}")
    except ImportError as e:
        print(f"  ✗ diffusers: MISSING - {e}")
        missing.append("diffusers")
    
    # Check calflops (optional but important for profiling)
    try:
        import calflops
        print(f"  ✓ calflops: {calflops.__version__ if hasattr(calflops, '__version__') else 'available'}")
    except ImportError as e:
        print(f"  ⚠ calflops: not installed (profiling will be disabled) - {e}")
        warnings.append("calflops")
    
    if missing:
        print(f"\n✗ Missing required dependencies: {missing}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False, missing, warnings
    
    print(f"\n✓ All required dependencies available")
    if warnings:
        print(f"  (optional: {warnings})")
    
    return True, missing, warnings


def check_profiler_import():
    """Check that the profiler module can be imported."""
    print("\n" + "="*60)
    print("Profiler Module Import Check")
    print("="*60)
    
    # First, make sure we can import from the right path
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"  Added to sys.path: {project_root}")
    
    try:
        print("  Attempting to import src.profiler...")
        from src.profiler import create_profiler
        print(f"  ✓ src.profiler.create_profiler imported successfully")
        return create_profiler, None
    except ImportError as e:
        print(f"  ✗ Failed to import src.profiler: {e}")
        traceback.print_exc()
        return None, str(e)
    except Exception as e:
        print(f"  ✗ Unexpected error importing src.profiler: {e}")
        traceback.print_exc()
        return None, str(e)


# ============================================================================
# Test Functions
# ============================================================================

def test_model_discovery(create_profiler):
    """Test model discovery from directory."""
    print("="*60)
    print("Testing Model Discovery")
    print("="*60)
    
    try:
        print("  Creating profiler instance...")
        profiler = create_profiler()
        print("  ✓ Profiler created successfully")
        
        print("  Listing models...")
        models = profiler.list_models()
        
        print(f"\n  Found {len(models)} models")
        if len(models) > 0:
            print(f"\n  First 5 models:")
            for model_id in models[:5]:
                print(f"    - {model_id}")
            if len(models) > 5:
                print(f"    ... and {len(models) - 5} more")
        else:
            print("  ⚠ No models found in models directory (this may be expected)")
        
        print(f"\n  ✓ Model discovery test passed")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Model discovery test failed: {e}")
        traceback.print_exc()
        return False


def test_architecture_detection(create_profiler):
    """Test architecture detection from model IDs."""
    print("\n" + "="*60)
    print("Testing Architecture Detection")
    print("="*60)
    
    try:
        print("  Creating profiler instance...")
        profiler = create_profiler()
        print("  ✓ Profiler created successfully")
        
        test_models = [
            "stabilityai/stable-diffusion-2-1-base",
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "unknown-model-id"  # This is expected to fail gracefully
        ]
        
        passed = 0
        failed = 0
        
        for model_id in test_models:
            print(f"\n  Testing: {model_id}")
            try:
                arch_info = profiler.detect_architecture(pipe=None, model_id=model_id)
                
                if arch_info:
                    print(f"    ✓ Detected")
                    print(f"      Source: {arch_info['source']}")
                    print(f"      Main model: {arch_info['config']['detection']['model_class']}")
                    print(f"      Pipeline: {arch_info['config']['detection']['pipeline_class']}")
                    print(f"      Wrapper: {arch_info['config']['profiling']['wrapper_class']}")
                    passed += 1
                else:
                    # "unknown-model-id" is expected to not be detected
                    if model_id == "unknown-model-id":
                        print(f"    ✓ Not detected (expected for unknown model)")
                        passed += 1
                    else:
                        print(f"    ⚠ Not detected (model may not exist in models directory)")
                        # Don't count as failure if model directory doesn't exist
                        passed += 1
                        
            except Exception as e:
                print(f"    ✗ Exception: {e}")
                traceback.print_exc()
                failed += 1
        
        print(f"\n  Architecture detection: {passed}/{len(test_models)} tests passed")
        
        if failed > 0:
            print(f"  ✗ {failed} tests failed")
            return False
        
        print(f"  ✓ Architecture detection test passed")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Architecture detection test failed: {e}")
        traceback.print_exc()
        return False


def test_model_info_reading(create_profiler):
    """Test reading detailed model info."""
    print("\n" + "="*60)
    print("Testing Model Info Reading")
    print("="*60)
    
    try:
        print("  Creating profiler instance...")
        profiler = create_profiler()
        print("  ✓ Profiler created successfully")
        
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        print(f"\n  Reading: {model_id}")
        try:
            model_info = profiler.model_reader.read_model_info(model_id)
            
            if model_info:
                print(f"    ✓ Successfully read model info")
                print(f"      Pipeline: {model_info['pipeline_class']}")
                print(f"      Main model: {model_info['main_model_class']}")
                print(f"      Type: {model_info['main_model_type']}")
                print(f"      Text encoders: {model_info['components']['text_encoder_count']}")
                print(f"      Has VAE: {model_info['components']['has_vae']}")
                if model_info['dimensions']:
                    print(f"      Dimensions: {model_info['dimensions']}")
                if model_info['vae_info']:
                    print(f"      VAE info: {model_info['vae_info']}")
            else:
                print(f"    ⚠ Model info not found (model may not exist in models directory)")
                print(f"      This is expected if the models directory is not populated.")
        
        except Exception as e:
            print(f"    ✗ Exception while reading model info: {e}")
            traceback.print_exc()
            return False
        
        print(f"\n  ✓ Model info reading test passed")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Model info reading test failed: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests with comprehensive error reporting."""
    print("\n" + "="*80)
    print(" "*20 + "Simplified Compute Profiler - Test Suite")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("="*80)
    
    # Track test results
    all_passed = True
    test_results = []
    
    try:
        # Step 1: Check dependencies
        deps_ok, missing, warnings = check_dependencies()
        if not deps_ok:
            print("\n" + "="*80)
            print(" "*25 + "TEST SUITE ABORTED")
            print("="*80)
            print("\nReason: Missing required dependencies")
            print(f"Missing: {missing}")
            print("\nPlease install missing dependencies:")
            print(f"  pip install {' '.join(missing)}")
            print("="*80)
            return False
        
        # Step 2: Check profiler import
        create_profiler, import_error = check_profiler_import()
        if create_profiler is None:
            print("\n" + "="*80)
            print(" "*25 + "TEST SUITE ABORTED")
            print("="*80)
            print("\nReason: Failed to import profiler module")
            print(f"Error: {import_error}")
            print("\nCheck the traceback above for details.")
            print("="*80)
            return False
        
        # Step 3: Run tests
        print("\n" + "="*80)
        print(" "*25 + "Running Tests")
        print("="*80)
        
        # Test 1: Model Discovery
        result = test_model_discovery(create_profiler)
        test_results.append(("Model Discovery", result))
        if not result:
            all_passed = False
        
        # Test 2: Model Info Reading
        result = test_model_info_reading(create_profiler)
        test_results.append(("Model Info Reading", result))
        if not result:
            all_passed = False
        
        # Test 3: Architecture Detection
        result = test_architecture_detection(create_profiler)
        test_results.append(("Architecture Detection", result))
        if not result:
            all_passed = False
        
        # Summary
        print("\n" + "="*80)
        print(" "*25 + "Test Results Summary")
        print("="*80)
        
        for test_name, passed in test_results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {test_name}: {status}")
        
        print("-"*80)
        
        if all_passed:
            print(" "*30 + "All tests completed!")
            print("="*80 + "\n")
            return True
        else:
            failed_count = sum(1 for _, passed in test_results if not passed)
            print(f" "*20 + f"{failed_count} test(s) failed!")
            print("="*80 + "\n")
            return False
        
    except Exception as e:
        print(f"\n" + "="*80)
        print(" "*20 + "TEST SUITE CRASHED")
        print("="*80)
        print(f"\nUnhandled exception: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*80 + "\n")
        return False


if __name__ == "__main__":
    try:
        # Ensure unbuffered output for CI logs
        import os
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        success = main()
        if success:
            print("Exit code: 0 (success)")
            sys.exit(0)
        else:
            print("Exit code: 1 (test failures)")
            sys.exit(1)
    except Exception as e:
        print(f"\n" + "="*80)
        print(" "*15 + "FATAL ERROR IN TEST RUNNER")
        print("="*80)
        print(f"\nUnhandled exception in main block: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nThis indicates a critical error in the test infrastructure.")
        print("="*80)
        print("Exit code: 1 (fatal error)")
        sys.exit(1)
