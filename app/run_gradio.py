#!/usr/bin/env python3
"""
Launcher script for the Gradio Documentation Agent
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
current_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

try:
    from app_gradio import create_interface

    if __name__ == "__main__":
        print("🚀 Starting LLM Documentation Agent with Gradio...")
        print("📍 Open your browser to: http://localhost:7860")

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
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting application: {e}")
    sys.exit(1)
