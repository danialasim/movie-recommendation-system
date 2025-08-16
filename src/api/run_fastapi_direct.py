#!/usr/bin/env python3
"""
Direct FastAPI runner from the API directory.
"""

import os
import sys
from pathlib import Path

# Change to the API directory
api_dir = Path(__file__).parent / "src" / "api"
os.chdir(api_dir)

# Add the API directory to Python path
sys.path.insert(0, str(api_dir))
sys.path.insert(0, str(api_dir.parent))

def run_fastapi_direct():
    """Run FastAPI directly from the API directory."""
    try:
        print("🚀 Starting FastAPI server from API directory...")
        print("📁 Working directory:", os.getcwd())
        print("🐍 Python path:", sys.path[:3])
        
        import uvicorn
        
        # Run the server
        uvicorn.run(
            "fastapi_app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Error starting FastAPI server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_fastapi_direct()
