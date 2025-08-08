"""Enhanced FastAPI application with OpenAI and Ollama support."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict
import os
import logging
from dotenv import load_dotenv
import uuid
from booking_client import BookingAPIClient
from agent import BookingAgent
import uvicorn
import requests

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TheHungryUnicorn Booking Agent", 
    version="2.0.0",
    description="AI-powered restaurant booking assistant with OpenAI and Ollama support"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the booking client
api_client = BookingAPIClient(
    base_url=os.getenv("BOOKING_API_URL", "http://localhost:8547"),
    bearer_token=os.getenv("BOOKING_API_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8"),
    restaurant_name="TheHungryUnicorn"
)

# Session management
sessions: Dict[str, BookingAgent] = {}


def get_llm_config():
    """Get LLM configuration from environment variables."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "ollama":
        return {
            "llm_provider": "ollama",
            "llm_model": os.getenv("OLLAMA_MODEL", "llama2"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        }
    else:
        return {
            "llm_provider": "openai",
            "llm_model": os.getenv("OPENAI_MODEL", "gpt-4"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3"))
        }


def get_or_create_agent(session_id: str) -> BookingAgent:
    """Get existing agent or create new one for session."""
    if session_id not in sessions:
        logger.info(f"Creating new agent for session: {session_id}")
        llm_config = get_llm_config()
        sessions[session_id] = BookingAgent(
            api_client,
            **llm_config
        )
    return sessions[session_id]


def cleanup_old_sessions():
    """Clean up old sessions to prevent memory leaks."""
    max_sessions = 100
    if len(sessions) > max_sessions:
        keys_to_remove = list(sessions.keys())[:-max_sessions]
        for key in keys_to_remove:
            logger.info(f"Removing old session: {key}")
            del sessions[key]


def check_ollama_status():
    """Check if Ollama server is running."""
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "running": True,
                "models": [model.get("name") for model in models]
            }
    except:
        pass
    return {"running": False, "models": []}


def check_openai_status():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return {
        "configured": bool(api_key and api_key != "your_openai_api_key_here"),
        "key_prefix": api_key[:8] + "..." if api_key else "Not set"
    }


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ConfigUpdate(BaseModel):
    llm_provider: str
    llm_model: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process a chat message and return the agent's response."""
    try:
        # Generate or use existing session ID
        session_id = message.session_id or str(uuid.uuid4())
        
        # Clean up old sessions periodically
        cleanup_old_sessions()
        
        # Get or create agent for this session
        agent = get_or_create_agent(session_id)
        
        # Process the message
        logger.info(f"Processing message for session {session_id}: {message.message[:50]}...")
        response = agent.process_message(message.message, session_id)
        
        logger.info(f"Response generated for session {session_id}")
        
        return ChatResponse(response=response, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        
        # Provide helpful error messages
        error_message = "Error processing message"
        if "ollama" in str(e).lower():
            error_message = "Ollama server is not running. Please start it with 'ollama serve'"
        elif "api" in str(e).lower() and "key" in str(e).lower():
            error_message = "OpenAI API key is not configured or invalid"
        
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a conversation session."""
    if session_id in sessions:
        logger.info(f"Resetting session: {session_id}")
        sessions[session_id].clear_memory(session_id)
        del sessions[session_id]
        return {"message": "Session reset successfully", "session_id": session_id}
    return {"message": "Session not found", "session_id": session_id}


@app.post("/config/update")
async def update_config(config: ConfigUpdate):
    """Update LLM configuration (affects new sessions only)."""
    os.environ["LLM_PROVIDER"] = config.llm_provider
    
    if config.llm_provider == "ollama" and config.llm_model:
        os.environ["OLLAMA_MODEL"] = config.llm_model
    elif config.llm_provider == "openai" and config.llm_model:
        os.environ["OPENAI_MODEL"] = config.llm_model
    
    return {
        "message": "Configuration updated for new sessions",
        "provider": config.llm_provider,
        "model": config.llm_model
    }


@app.get("/config/status")
async def get_config_status():
    """Get current LLM configuration and status."""
    current_config = get_llm_config()
    ollama_status = check_ollama_status()
    openai_status = check_openai_status()
    
    return {
        "current_provider": current_config["llm_provider"],
        "current_model": current_config.get("llm_model"),
        "providers": {
            "openai": {
                "available": openai_status["configured"],
                "status": "Configured" if openai_status["configured"] else "API key not set",
                "api_key": openai_status["key_prefix"]
            },
            "ollama": {
                "available": ollama_status["running"],
                "status": "Running" if ollama_status["running"] else "Server not running",
                "models": ollama_status["models"],
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            }
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check booking API
    try:
        result = api_client.check_availability("2025-08-10")
        api_status = "connected" if result.get('success') else "error"
    except Exception as e:
        api_status = f"error: {str(e)}"
    
    # Check LLM provider
    llm_config = get_llm_config()
    llm_status = "configured"
    
    if llm_config["llm_provider"] == "ollama":
        ollama_status = check_ollama_status()
        if not ollama_status["running"]:
            llm_status = "Ollama server not running"
    else:
        openai_status = check_openai_status()
        if not openai_status["configured"]:
            llm_status = "OpenAI API key not configured"
    
    return {
        "status": "healthy",
        "booking_api": api_status,
        "llm_provider": llm_config["llm_provider"],
        "llm_status": llm_status,
        "active_sessions": len(sessions)
    }


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Frontend not found. Please ensure frontend/index.html exists."}


@app.get("/stats")
async def get_stats():
    """Get application statistics."""
    return {
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys()),
        "llm_provider": get_llm_config()["llm_provider"]
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 60)
    logger.info("Starting TheHungryUnicorn Booking Agent...")
    logger.info("=" * 60)
    
    # Log configuration
    llm_config = get_llm_config()
    logger.info(f"LLM Provider: {llm_config['llm_provider']}")
    logger.info(f"LLM Model: {llm_config.get('llm_model')}")
    logger.info(f"Booking API URL: {os.getenv('BOOKING_API_URL', 'http://localhost:8547')}")
    
    # Check LLM availability
    if llm_config["llm_provider"] == "ollama":
        ollama_status = check_ollama_status()
        if ollama_status["running"]:
            logger.info(f"✅ Ollama server is running")
            logger.info(f"   Available models: {', '.join(ollama_status['models'])}")
        else:
            logger.warning("⚠️ Ollama server is not running. Start it with: ollama serve")
    else:
        openai_status = check_openai_status()
        if openai_status["configured"]:
            logger.info(f"✅ OpenAI API key configured: {openai_status['key_prefix']}")
        else:
            logger.warning("⚠️ OpenAI API key not configured. Set OPENAI_API_KEY in .env file")
    
    # Test booking API connection
    try:
        result = api_client.check_availability("2025-08-10")
        if result.get('success'):
            logger.info("✅ Successfully connected to booking API")
        else:
            logger.warning("⚠️ Connected to booking API but received error response")
    except Exception as e:
        logger.error(f"❌ Failed to connect to booking API: {e}")
    
    logger.info("=" * 60)
    logger.info("Server ready at http://localhost:8000")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down TheHungryUnicorn Booking Agent...")
    logger.info(f"Clearing {len(sessions)} active sessions...")
    sessions.clear()


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )