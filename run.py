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
  python run.py              # Launch user mode (default)
  python run.py --dev        # Launch developer mode
  python run.py --help       # Show this help message
        """
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Launch in developer mode with manual model selection"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (default: 7860 for user mode, 7861 for dev mode)"
    )
    
    args = parser.parse_args()
    
    if args.dev:
        print("🔧 Launching Developer Mode...")
        from src.app_dev import main as dev_main
        if args.port:
            import src.app_dev as app_dev
            app_dev.main()
        else:
            dev_main()
    else:
        print("🚀 Launching User Mode...")
        from src.app import main as app_main
        if args.port:
            import src.app as app
            app.main()
        else:
            app_main()


if __name__ == "__main__":
    main()

