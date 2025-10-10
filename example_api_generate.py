#!/usr/bin/env python3
"""
Example script demonstrating how to use closed-source API clients.
"""

import os
from src.api_clients import get_api_client, OpenAIClient, GoogleImagenClient, BytedanceClient, KlingClient
from PIL import Image


def example_openai():
    """Example: Generate with OpenAI DALL-E."""
    print("\n" + "="*60)
    print("Example 1: OpenAI DALL-E")
    print("="*60)
    
    # Option 1: Use the factory function
    client = get_api_client("openai")
    
    # Option 2: Create client directly
    # client = OpenAIClient(api_key="your-api-key")
    
    if not client:
        print("❌ Failed to create OpenAI client")
        return
    
    prompt = "A serene Japanese garden with a red bridge over a koi pond"
    
    print(f"\nPrompt: {prompt}")
    print("Generating with DALL-E 3...")
    
    image, error = client.generate(
        prompt=prompt,
        width=1024,
        height=1024,
        model="dall-e-3",
        quality="standard",  # or "hd"
        style="vivid",       # or "natural"
    )
    
    if image:
        print("✅ Success!")
        image.save("example_openai.png")
        print("Saved to: example_openai.png")
    else:
        print(f"❌ Error: {error}")


def example_google():
    """Example: Generate with Google Imagen."""
    print("\n" + "="*60)
    print("Example 2: Google Imagen")
    print("="*60)
    
    client = get_api_client("google")
    
    if not client:
        print("❌ Failed to create Google Imagen client")
        return
    
    prompt = "A futuristic cityscape at sunset with flying cars"
    
    print(f"\nPrompt: {prompt}")
    print("Generating with Google Imagen...")
    
    image, error = client.generate(
        prompt=prompt,
        width=1024,
        height=1024,
    )
    
    if image:
        print("✅ Success!")
        image.save("example_google.png")
        print("Saved to: example_google.png")
    else:
        print(f"❌ Error: {error}")


def example_bytedance():
    """Example: Generate with Bytedance."""
    print("\n" + "="*60)
    print("Example 3: Bytedance Cloud")
    print("="*60)
    
    client = get_api_client("bytedance")
    
    if not client:
        print("❌ Failed to create Bytedance client")
        return
    
    prompt = "A magical forest with glowing mushrooms and fireflies"
    
    print(f"\nPrompt: {prompt}")
    print("Generating with Bytedance API...")
    
    image, error = client.generate(
        prompt=prompt,
        width=1024,
        height=1024,
    )
    
    if image:
        print("✅ Success!")
        image.save("example_bytedance.png")
        print("Saved to: example_bytedance.png")
    else:
        print(f"❌ Error: {error}")


def example_kling():
    """Example: Generate with Kling AI."""
    print("\n" + "="*60)
    print("Example 4: Kling AI")
    print("="*60)
    
    client = get_api_client("kling")
    
    if not client:
        print("❌ Failed to create Kling client")
        return
    
    prompt = "A majestic dragon flying over snow-capped mountains"
    
    print(f"\nPrompt: {prompt}")
    print("Generating with Kling AI...")
    
    image, error = client.generate(
        prompt=prompt,
        width=1024,
        height=1024,
        model="kling-v1",  # or "kling-v1-pro"
    )
    
    if image:
        print("✅ Success!")
        image.save("example_kling.png")
        print("Saved to: example_kling.png")
    else:
        print(f"❌ Error: {error}")


def example_batch_comparison():
    """Example: Compare all APIs with the same prompt."""
    print("\n" + "="*60)
    print("Example 5: Batch Comparison")
    print("="*60)
    
    prompt = "A beautiful sunset over the ocean with palm trees"
    print(f"\nPrompt: {prompt}")
    print("Generating with all available APIs...\n")
    
    providers = ["openai", "google", "bytedance", "kling"]
    results = []
    
    for provider in providers:
        print(f"🔄 Processing {provider}...")
        client = get_api_client(provider)
        
        if not client:
            print(f"  ⚠️  Skipping {provider} (no API key configured)")
            continue
        
        image, error = client.generate(
            prompt=prompt,
            width=1024,
            height=1024,
        )
        
        if image:
            filename = f"comparison_{provider}.png"
            image.save(filename)
            print(f"  ✅ Success! Saved to {filename}")
            results.append((provider, image))
        else:
            print(f"  ❌ Failed: {error}")
    
    print(f"\n📊 Generated {len(results)} images successfully!")


def main():
    """Main function to run examples."""
    print("="*60)
    print("Closed-Source API Image Generation Examples")
    print("="*60)
    
    # Check for API keys
    print("\nChecking API keys...")
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID"),
        "BYTEDANCE_API_KEY": os.getenv("BYTEDANCE_API_KEY"),
        "KLING_API_KEY": os.getenv("KLING_API_KEY"),
    }
    
    configured = []
    missing = []
    
    for key, value in api_keys.items():
        if value:
            configured.append(key)
        else:
            missing.append(key)
    
    if configured:
        print(f"✅ Configured: {', '.join(configured)}")
    if missing:
        print(f"⚠️  Missing: {', '.join(missing)}")
    
    if not configured:
        print("\n❌ No API keys configured!")
        print("Please set environment variables or edit .env file.")
        print("See .env.example for details.")
        return
    
    print("\nSelect an example to run:")
    print("  1. OpenAI DALL-E")
    print("  2. Google Imagen")
    print("  3. Bytedance Cloud")
    print("  4. Kling AI")
    print("  5. Batch Comparison (all APIs)")
    print("  6. Run all examples")
    print("  0. Exit")
    
    choice = input("\nEnter choice (0-6): ").strip()
    
    examples = {
        "1": example_openai,
        "2": example_google,
        "3": example_bytedance,
        "4": example_kling,
        "5": example_batch_comparison,
    }
    
    if choice == "0":
        print("Goodbye!")
        return
    elif choice == "6":
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"❌ Error: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Invalid choice!")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
