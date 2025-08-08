#!/usr/bin/env python3
"""
Setup script for TheHungryUnicorn Booking Agent
Helps configure the environment and check dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")


def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.8+ required (found {version.major}.{version.minor})")
        return False


def check_pip():
    """Check if pip is installed"""
    if shutil.which("pip") or shutil.which("pip3"):
        print_success("pip is installed")
        return True
    else:
        print_error("pip is not installed")
        return False


def install_requirements():
    """Install Python requirements"""
    print_info("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print_success("Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e.stderr}")
        return False


def check_ollama():
    """Check if Ollama is installed and running"""
    ollama_installed = shutil.which("ollama") is not None
    
    if ollama_installed:
        print_success("Ollama is installed")
        
        # Check if Ollama server is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                print_success(f"Ollama server is running ({len(models)} models available)")
                if models:
                    print_info("Available models: " + ", ".join([m['name'] for m in models]))
                return True
            else:
                print_warning("Ollama installed but server not running")
                print_info("Start it with: ollama serve")
                return False
        except:
            print_warning("Ollama installed but server not running")
            print_info("Start it with: ollama serve")
            return False
    else:
        print_warning("Ollama not installed (optional for local LLM support)")
        print_info("Install from: https://ollama.ai/download")
        return False


def setup_env_file():
    """Create .env file from template"""
    env_path = Path("backend/.env")
    env_example_path = Path("backend/.env.example")
    
    if env_path.exists():
        print_info(".env file already exists")
        response = input("Do you want to reconfigure it? (y/n): ").lower()
        if response != 'y':
            return True
    
    print_info("Setting up environment configuration...")
    
    # Ask for LLM provider preference
    print("\nWhich LLM provider would you like to use?")
    print("1. OpenAI (requires API key, better quality)")
    print("2. Ollama (free, local, requires Ollama installation)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    env_content = """# Environment Configuration for TheHungryUnicorn Booking Agent

# Booking API Configuration
BOOKING_API_URL=http://localhost:8547
BOOKING_API_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8

# LLM Temperature
LLM_TEMPERATURE=0.3

"""
    
    if choice == "1":
        # OpenAI setup
        api_key = input("Enter your OpenAI API key (or press Enter to add later): ").strip()
        if not api_key:
            api_key = "your_openai_api_key_here"
            print_warning("Remember to add your OpenAI API key to .env file")
        
        env_content += f"""# LLM Provider
LLM_PROVIDER=openai

# OpenAI Configuration
OPENAI_API_KEY={api_key}
OPENAI_MODEL=gpt-4

# Ollama Configuration (not used)
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2:3b
"""
    else:
        # Ollama setup
        env_content += f"""# LLM Provider
LLM_PROVIDER=ollama

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# OpenAI Configuration (not used)
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4
"""
        
        if not check_ollama():
            print_warning("Ollama is not running. Remember to:")
            print_info("1. Install Ollama: https://ollama.ai/download")
            print_info("2. Start server: ollama serve")
            print_info("3. Pull a model: ollama pull llama3.2:3b")
    
    # Write the env file
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print_success(f".env file created at {env_path}")
    return True


def check_booking_api():
    """Check if the mock booking API is running"""
    try:
        import requests
        response = requests.get("http://localhost:8547/docs", timeout=2)
        if response.status_code == 200:
            print_success("Mock Booking API is running on port 8547")
            return True
    except:
        pass
    
    print_warning("Mock Booking API is not running")
    print_info("Start it with: cd Restaurant-Booking-Mock-API-Server && python -m app")
    return False


def main():
    print_header("TheHungryUnicorn Booking Agent Setup")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print_error("Failed to install dependencies")
        sys.exit(1)
    
    # Setup environment file
    setup_env_file()
    
    # Check optional services
    print_header("Checking Services")
    
    ollama_ok = check_ollama()
    booking_api_ok = check_booking_api()
    
    # Final instructions
    print_header("Setup Complete!")
    
    print_info("Next steps:")
    print(f"1. {'✅' if booking_api_ok else '⚠️ '} Start Mock Booking API: cd Restaurant-Booking-Mock-API-Server && python -m app")
    
    with open("backend/.env", 'r') as f:
        env_content = f.read()
        if "LLM_PROVIDER=ollama" in env_content:
            print(f"2. {'✅' if ollama_ok else '⚠️ '} Start Ollama: ollama serve")
            if not ollama_ok:
                print("   Then pull a model: ollama pull llama3.2:3b")
        else:
            if "your_openai_api_key_here" in env_content:
                print("2. ⚠️  Add your OpenAI API key to backend/.env")
    
    print("3. Start the agent: cd backend && python app.py")
    print("4. Open browser: http://localhost:8000")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Happy booking! 🦄{Colors.END}")


if __name__ == "__main__":
    main()