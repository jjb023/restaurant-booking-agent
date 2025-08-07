from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from typing import Dict, Any, Optional
import logging
from booking_client import BookingAPIClient
from tools import (
    CheckAvailabilityTool, CreateBookingTool, GetBookingTool,
    UpdateBookingTool, CancelBookingTool
)

logger = logging.getLogger(__name__)


class BookingAgent:
    """Conversational agent for restaurant bookings using local LLM."""
    
    def __init__(self, api_client: BookingAPIClient, model_name: str = "mistral", temperature: float = 0.7):
        self.api_client = api_client
        
        # Use Ollama for local LLM (free!)
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434"  # Default Ollama URL
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = [
            CheckAvailabilityTool(api_client),
            CreateBookingTool(api_client),
            GetBookingTool(api_client),
            UpdateBookingTool(api_client),
            CancelBookingTool(api_client)
        ]
        
        # Create the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": """You are a helpful restaurant booking assistant for TheHungryUnicorn restaurant. 
                You help customers make, check, modify, and cancel their reservations.
                
                When helping customers:
                1. Be friendly and professional
                2. Ask for missing information when needed (name, date, time, party size for new bookings)
                3. Confirm important details before making changes
                4. Provide booking references after creating reservations
                5. Handle dates naturally (understand "next Friday", "this weekend", etc.)
                
                You have access to these tools:"""
            }
        )
        
        # Store session data
        self.session_data: Dict[str, Any] = {}
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """
        Process a user message and return the agent's response.
        
        Args:
            message: The user's message
            session_id: Optional session ID for maintaining conversation state
        
        Returns:
            The agent's response
        """
        try:
            # Store any extracted booking info in session
            if session_id and session_id not in self.session_data:
                self.session_data[session_id] = {}
            
            # Process the message
            response = self.agent.run(message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._get_fallback_response(message)
    
    def _get_fallback_response(self, message: str) -> str:
        """Provide helpful fallback responses based on intent."""
        message_lower = message.lower()
        
        if "availability" in message_lower or "available" in message_lower:
            return "I can help you check availability. Please specify a date (like 'this weekend' or 'next Friday') and I'll check what's available."
        elif "book" in message_lower or "reservation" in message_lower:
            return "I'd be happy to help you make a reservation. Please provide: your name, the date, time, and number of people."
        elif "cancel" in message_lower:
            return "I can help you cancel a reservation. Please provide your booking reference number (it starts with 'BK')."
        elif "change" in message_lower or "modify" in message_lower:
            return "I can help you modify your reservation. Please provide your booking reference and what you'd like to change."
        else:
            return "I'm here to help with restaurant bookings. How can I assist you today?"
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session."""
        self.memory.clear()
        if session_id and session_id in self.session_data:
            del self.session_data[session_id]