#!/usr/bin/env python3
"""Setup script for the Ollama-based restaurant booking agent."""

import os
import sys
import subprocess
import requests
from pathlib import Path

def check_ollama_installation():
    """Check if Ollama is installed and running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running on http://localhost:11434")
            return True
        else:
            print("❌ Ollama is not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running. Please start Ollama first.")
        print("   You can download it from: https://ollama.ai")
        return False

def check_model_availability(model_name="llama3.2:3b"):
    """Check if the required model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            if model_name in available_models:
                print(f"✅ Model {model_name} is available")
                return True
            else:
                print(f"⚠️  Model {model_name} is not available")
                print(f"   Available models: {', '.join(available_models)}")
                print(f"   You can pull the model with: ollama pull {model_name}")
                return False
    except Exception as e:
        print(f"❌ Error checking model availability: {e}")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_env_file():
    """Create a .env file with default configuration."""
    env_content = """# Restaurant Booking Agent Configuration

# Ollama Configuration
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

# API Configuration
BOOKING_API_URL=http://localhost:8547
BOOKING_API_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8

# Server Configuration
HOST=0.0.0.0
PORT=8000
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
    else:
        print("ℹ️  .env file already exists")

def run_tests():
    """Run basic tests to verify the setup."""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        from booking_client import BookingAPIClient
        from agent_ollama import BookingAgent
        from tools import CheckAvailabilityTool
        print("✅ All imports successful")
        
        # Test API client
        api_client = BookingAPIClient(
            base_url="http://localhost:8547",
            bearer_token="test-token",
            restaurant_name="TheHungryUnicorn"
        )
        print("✅ API client initialized")
        
        # Test agent initialization
        agent = BookingAgent(api_client)
        print("✅ Agent initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Ollama-based Restaurant Booking Agent")
    print("=" * 50)
    
    # Check Ollama
    if not check_ollama_installation():
        print("\n📋 To install Ollama:")
        print("1. Visit https://ollama.ai")
        print("2. Download and install Ollama")
        print("3. Start Ollama")
        print("4. Run: ollama pull llama3.2:3b")
        return False
    
    # Check model
    if not check_model_availability():
        print(f"\n📋 To install the model:")
        print(f"Run: ollama pull llama3.2:3b")
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create env file
    create_env_file()
    
    # Run tests
    if not run_tests():
        return False
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 To run the server:")
    print("python app.py")
    print("\n📋 To test the API:")
    print("curl -X POST http://localhost:8000/chat \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"message\": \"Hello\"}'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
