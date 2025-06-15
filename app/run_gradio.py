#!/usr/bin/env python3
"""
Main launcher for LLM Documentation Agent
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path for robust imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Launch the Gradio application"""
    try:
        # Import and run the Gradio app
        from app.app_gradio import create_interface

        print("🚀 Starting LLM Documentation Agent with Gradio...")
        print("📍 Open your browser to: http://localhost:7860")
        print("🗂️ Project structure reorganized for better maintainability")
        print("🔑 API keys will be loaded from your .env file")
        print()

        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=True,
            inbrowser=True,  # Automatically open browser
        )

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
        print("📁 Current working directory:", os.getcwd())
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
