"""FastAPI application for the booking agent using Ollama."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
from dotenv import load_dotenv
import uuid
from booking_client import BookingAPIClient
from agent_ollama import BookingAgent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Restaurant Booking Agent (Ollama)", version="2.0.0")

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
    bearer_token=os.getenv("BOOKING_API_TOKEN", "test_token"),
    restaurant_name="TheHungryUnicorn"
)

# Model configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize the agent with Ollama
logger.info(f"Initializing BookingAgent with model: {MODEL_NAME}")
try:
    booking_agent = BookingAgent(
        api_client,
        model_name=MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )
    logger.info("BookingAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize BookingAgent: {e}")
    raise

sessions = {}


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    model: str = MODEL_NAME


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process a chat message and return the agent's response."""
    try:
        session_id = message.session_id or str(uuid.uuid4())
        
        # Create or retrieve session agent
        if session_id not in sessions:
            logger.info(f"Creating new session: {session_id}")
            sessions[session_id] = BookingAgent(
                api_client,
                model_name=MODEL_NAME,
                base_url=OLLAMA_BASE_URL
            )
        
        agent = sessions[session_id]
        
        logger.info(f"Processing message for session {session_id}: {message.message}")
        response = agent.process_message(message.message, session_id)
        logger.info(f"Response generated: {response[:100]}...")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            model=MODEL_NAME
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        # Return a helpful error message instead of raising
        return ChatResponse(
            response=f"I apologize, but I encountered an error. Please make sure Ollama is running and the model {MODEL_NAME} is installed. You can install it with: ollama pull {MODEL_NAME}",
            session_id=message.session_id or str(uuid.uuid4()),
            model=MODEL_NAME
        )


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a conversation session."""
    if session_id in sessions:
        sessions[session_id].clear_memory(session_id)
        logger.info(f"Session reset: {session_id}")
        return {"message": "Session reset successfully"}
    return {"message": "Session not found"}


@app.get("/health")
async def health_check():
    """Health check endpoint with Ollama status."""
    import requests
    
    health_status = {
        "status": "healthy",
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_BASE_URL
    }
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            health_status["ollama_status"] = "connected"
            health_status["available_models"] = models
            health_status["model_installed"] = MODEL_NAME in models
            
            if not health_status["model_installed"]:
                health_status["warning"] = f"Model {MODEL_NAME} not installed. Run: ollama pull {MODEL_NAME}"
        else:
            health_status["ollama_status"] = "error"
            health_status["error"] = f"Ollama returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        health_status["ollama_status"] = "disconnected"
        health_status["error"] = "Cannot connect to Ollama. Make sure it's running with: ollama serve"
    except Exception as e:
        health_status["ollama_status"] = "error"
        health_status["error"] = str(e)
    
    return health_status


@app.get("/models")
async def list_models():
    """List available Ollama models."""
    import requests
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [
                {
                    "name": model['name'],
                    "size": f"{model.get('size', 0) / 1e9:.1f}GB" if model.get('size') else "Unknown",
                    "modified": model.get('modified_at', 'Unknown')
                }
                for model in data.get('models', [])
            ]
            return {
                "current_model": MODEL_NAME,
                "models": models,
                "total": len(models)
            }
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
    
    return {"error": "Could not fetch models from Ollama", "ollama_url": OLLAMA_BASE_URL}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Restaurant Booking Agent API",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "models": "/models",
            "reset": "/reset/{session_id}",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server with Ollama model: {MODEL_NAME}")
    logger.info(f"Ollama URL: {OLLAMA_BASE_URL}")
    logger.info("Make sure Ollama is running: ollama serve")
    logger.info(f"Make sure model is installed: ollama pull {MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=8000)