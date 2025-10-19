"""
Test script for the simplified compute profiler.
Run from project root: python -m src.profiler.test_profiler
"""

from .profiler import create_profiler

def test_model_discovery():
    """Test model discovery from directory."""
    print("="*60)
    print("Testing Model Discovery")
    print("="*60)
    
    profiler = create_profiler()
    models = profiler.list_models()
    
    print(f"\nFound {len(models)} models")
    print(f"\nFirst 5 models:")
    for model_id in models[:5]:
        print(f"  - {model_id}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")

def test_architecture_detection():
    """Test architecture detection from model IDs."""
    print("\n" + "="*60)
    print("Testing Architecture Detection")
    print("="*60)
    
    profiler = create_profiler()
    
    test_models = [
        "stabilityai/stable-diffusion-2-1-base",
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "unknown-model-id"
    ]
    
    for model_id in test_models:
        print(f"\nTesting: {model_id}")
        arch_info = profiler.detect_architecture(pipe=None, model_id=model_id)
        
        if arch_info:
            print(f"  ✓ Detected")
            print(f"    Source: {arch_info['source']}")
            print(f"    Main model: {arch_info['config']['detection']['model_class']}")
            print(f"    Pipeline: {arch_info['config']['detection']['pipeline_class']}")
            print(f"    Wrapper: {arch_info['config']['profiling']['wrapper_class']}")
        else:
            print(f"  ✗ Not detected")

def test_model_info_reading():
    """Test reading detailed model info."""
    print("\n" + "="*60)
    print("Testing Model Info Reading")
    print("="*60)
    
    profiler = create_profiler()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    print(f"\nReading: {model_id}")
    model_info = profiler.model_reader.read_model_info(model_id)
    
    if model_info:
        print(f"  ✓ Successfully read model info")
        print(f"    Pipeline: {model_info['pipeline_class']}")
        print(f"    Main model: {model_info['main_model_class']}")
        print(f"    Type: {model_info['main_model_type']}")
        print(f"    Text encoders: {model_info['components']['text_encoder_count']}")
        print(f"    Has VAE: {model_info['components']['has_vae']}")
        if model_info['dimensions']:
            print(f"    Dimensions: {model_info['dimensions']}")
        if model_info['vae_info']:
            print(f"    VAE info: {model_info['vae_info']}")
    else:
        print(f"  ✗ Failed to read model info")

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" "*20 + "Simplified Compute Profiler - Test Suite")
    print("="*80)
    
    try:
        test_model_discovery()
        test_model_info_reading()
        test_architecture_detection()
        
        print("\n" + "="*80)
        print(" "*30 + "All tests completed!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
