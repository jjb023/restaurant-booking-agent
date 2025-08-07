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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the simple agent by default (more reliable with Ollama)
logger.info("Using SimpleBookingAgent (more reliable with Ollama)")
from agent_simple import SimpleBookingAgent as BookingAgent

# Initialize FastAPI app
app = FastAPI(title="Restaurant Booking Agent (Ollama)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the booking client with correct token
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8"

api_client = BookingAPIClient(
    base_url=os.getenv("BOOKING_API_URL", "http://localhost:8547"),
    bearer_token=os.getenv("BOOKING_API_TOKEN", BEARER_TOKEN),
    restaurant_name="TheHungryUnicorn"
)

# Model configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Test Ollama connection
def test_ollama_connection():
    """Test if Ollama is running and model is available."""
    import requests
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code != 200:
            return False, "Ollama API error"
        
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        
        if not models:
            return False, f"No models installed. Run: ollama pull {MODEL_NAME}"
        
        if MODEL_NAME not in models:
            # Try without the tag
            model_base = MODEL_NAME.split(':')[0]
            if not any(model_base in m for m in models):
                return False, f"Model {MODEL_NAME} not installed. Run: ollama pull {MODEL_NAME}"
        
        logger.info(f"✅ Ollama connected with model {MODEL_NAME}")
        return True, "Connected"
        
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Run: ollama serve"
    except Exception as e:
        return False, str(e)

# Test connection on startup
ollama_ok, ollama_message = test_ollama_connection()
if not ollama_ok:
    logger.warning(f"⚠️  Ollama issue: {ollama_message}")

# Initialize the global agent
try:
    logger.info(f"Initializing agent with model: {MODEL_NAME}")
    booking_agent = BookingAgent(
        api_client,
        model_name=MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )
    logger.info("✅ BookingAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    # Create fallback
    class FallbackAgent:
        def __init__(self, api_client):
            self.api_client = api_client
            self.conversations = {}
        
        def process_message(self, message: str, session_id: Optional[str] = None) -> str:
            msg_lower = message.lower()
            if 'book' in msg_lower or 'reservation' in msg_lower:
                return "I'd love to help you make a reservation! I'll need: your name, preferred date, time, and party size."
            elif 'availability' in msg_lower:
                return "I can check availability for you. What date are you interested in?"
            elif 'cancel' in msg_lower:
                return "I can help cancel your reservation. Please provide your booking reference."
            else:
                return "Welcome to TheHungryUnicorn! I can help you make reservations, check availability, or manage existing bookings."
        
        def clear_memory(self, session_id: Optional[str] = None):
            pass
    
    booking_agent = FallbackAgent(api_client)

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
        
        logger.info(f"Processing: {message.message[:50]}...")
        response = agent.process_message(message.message, session_id)
        logger.info(f"Response: {response[:50]}...")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            model=MODEL_NAME
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return ChatResponse(
            response="I apologize for the technical issue. Please try again or check that all services are running.",
            session_id=message.session_id or str(uuid.uuid4()),
            model=MODEL_NAME
        )


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a conversation session."""
    if session_id in sessions:
        sessions[session_id].clear_memory(session_id)
        del sessions[session_id]
        logger.info(f"Session reset: {session_id}")
        return {"message": "Session reset successfully"}
    return {"message": "Session not found"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import requests
    
    # Check Ollama
    ollama_status = "unknown"
    models = []
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            ollama_status = "connected"
    except:
        ollama_status = "disconnected"
    
    # Check Mock API
    api_status = "unknown"
    try:
        response = requests.get(f"{api_client.base_url}/docs", timeout=2)
        api_status = "connected" if response.status_code == 200 else "disconnected"
    except:
        api_status = "disconnected"
    
    return {
        "status": "healthy",
        "services": {
            "ollama": {
                "status": ollama_status,
                "url": OLLAMA_BASE_URL,
                "model": MODEL_NAME,
                "models_installed": models
            },
            "booking_api": {
                "status": api_status,
                "url": api_client.base_url
            }
        },
        "configuration": {
            "agent_type": "simple",
            "model": MODEL_NAME
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Restaurant Booking Agent",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "reset": "POST /reset/{session_id}",
            "docs": "/docs"
        },
        "status": "Ready to help with restaurant bookings!"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("🍽️  Restaurant Booking Agent Starting...")
    print("="*50)
    print(f"📦 Model: {MODEL_NAME}")
    print(f"🔗 Ollama URL: {OLLAMA_BASE_URL}")
    print(f"🔗 Booking API: {api_client.base_url}")
    print("="*50)
    print("\n⚠️  Make sure these services are running:")
    print("1. Ollama: ollama serve")
    print(f"2. Model installed: ollama pull {MODEL_NAME}")
    print("3. Mock API: cd Restaurant-Booking-Mock-API-Server && python -m app")
    print("\n✅ Starting server on http://localhost:8000")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)