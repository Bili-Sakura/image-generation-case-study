#!/usr/bin/env python3
"""
Launcher script for the text-to-image generation application.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-Image Generation Case Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --dev        # Launch developer mode
  python run.py --bench      # Launch benchmark mode (all models)
  python run.py --help       # Show this help message
        """
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Launch in developer mode with manual model selection (port 7861)"
    )
    
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Launch in benchmark mode - systematically test all models with load-inference-unload cycle (port 7862)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (default: 7861 for dev mode, 7862 for bench mode)"
    )
    
    args = parser.parse_args()
    
    # Require explicit mode selection
    if not args.dev and not args.bench:
        parser.error("Please specify a mode: --dev or --bench")
    
    if args.bench:
        print("📊 Launching Benchmark Mode...")
        from src.app_bench import main as bench_main
        bench_main()
    elif args.dev:
        print("🔧 Launching Developer Mode...")
        from src.app_dev import main as dev_main
        dev_main()


if __name__ == "__main__":
    main()

