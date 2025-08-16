#!/usr/bin/env python3
"""
Test and run the FastAPI movie recommendation server.
"""

import sys
import subprocess
from pathlib import Path

def check_and_install_dependencies():
    """Check and install FastAPI dependencies."""
    required_packages = [
        'fastapi',
        'uvicorn[standard]',
        'pydantic'
    ]
    
    print("🔍 Checking FastAPI dependencies...")
    
    for package in required_packages:
        try:
            if package == 'uvicorn[standard]':
                import uvicorn
                print(f"✅ uvicorn is installed")
            elif package == 'fastapi':
                import fastapi
                print(f"✅ fastapi is installed")
            elif package == 'pydantic':
                import pydantic
                print(f"✅ pydantic is installed")
        except ImportError:
            print(f"❌ {package} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def test_fastapi_import():
    """Test if the FastAPI app can be imported."""
    try:
        # Add src to path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root / "src"))
        
        print("🧪 Testing FastAPI app import...")
        
        # Test import without running
        from src.api.fastapi_app import app
        print("✅ FastAPI app imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_fastapi_server():
    """Run the FastAPI server."""
    try:
        print("🚀 Starting FastAPI server...")
        print("📍 Server will be available at:")
        print("   - Main API: http://localhost:8000")
        print("   - Swagger Docs: http://localhost:8000/docs")
        print("   - ReDoc: http://localhost:8000/redoc")
        print("   - Health Check: http://localhost:8000/health")
        print("\n🛑 Press Ctrl+C to stop the server")
        
        # Import and run
        import uvicorn
        uvicorn.run(
            "src.api.fastapi_app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to test and run FastAPI server."""
    print("🎬 MovieRec AI - FastAPI Server Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_and_install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    print("\n" + "=" * 40)
    
    # Test import
    if not test_fastapi_import():
        print("❌ FastAPI app import failed")
        return
    
    print("\n" + "=" * 40)
    
    # Run server
    run_fastapi_server()

if __name__ == "__main__":
    main()
