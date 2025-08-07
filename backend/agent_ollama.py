"""LangChain agent implementation using Ollama for restaurant bookings."""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from typing import List, Dict, Any, Optional
import logging
from booking_client import BookingAPIClient
from tools import (
    CheckAvailabilityTool, CreateBookingTool, GetBookingTool,
    UpdateBookingTool, CancelBookingTool
)

logger = logging.getLogger(__name__)


class BookingAgent:
    """Conversational agent for restaurant bookings using Ollama."""
    
    def __init__(
        self, 
        api_client: BookingAPIClient, 
        model_name: str = "llama3.2:3b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434"
    ):
        self.api_client = api_client
        
        # Initialize Ollama LLM without the format parameter
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            num_predict=512,  # Max tokens to generate
            top_p=0.9,
            repeat_penalty=1.1  # Prevent repetition
        )
        
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
        
        # Create the agent with optimized prompt for Ollama
        self.agent_executor = self._create_agent()
        
        # Store session data
        self.session_data: Dict[str, Any] = {}
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with Ollama-optimized prompt."""
        
        # Simplified and clearer prompt for better Ollama performance
        prompt = PromptTemplate.from_template("""You are a helpful restaurant booking assistant for TheHungryUnicorn restaurant.

You have access to these tools:
{tools}

Use these tools to help customers:
- check_availability: Check if tables are available (needs: date, optional: time, party_size)
- create_booking: Make a new reservation (needs: customer_name, date, time, party_size)
- get_booking: Look up booking details (needs: booking_id)
- update_booking: Change a booking (needs: booking_id, and what to change)
- cancel_booking: Cancel a reservation (needs: booking_id)

Previous conversation:
{chat_history}

Customer: {input}

To respond, use this format:
Thought: I need to understand what the customer wants
Action: the tool to use
Action Input: the input for the tool
Observation: the result from the tool
... (repeat if needed)
Thought: Now I can respond to the customer
Final Answer: Your helpful response to the customer

Begin!
{agent_scratchpad}""")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """Process a user message and return the agent's response."""
        try:
            # Store any extracted booking info in session
            if session_id and session_id not in self.session_data:
                self.session_data[session_id] = {}
            
            # Run the agent with invoke instead of run
            response = self.agent_executor.invoke({"input": message})
            
            # Extract the output
            if isinstance(response, dict) and 'output' in response:
                return response['output']
            return str(response)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Fallback to direct response if agent fails
            return self._fallback_response(message)
    
    def _fallback_response(self, message: str) -> str:
        """Fallback to direct LLM response if agent fails."""
        try:
            # Simple direct response without agent framework
            prompt = f"""You are a helpful restaurant booking assistant for TheHungryUnicorn.
            
Customer message: {message}

Provide a helpful response. If they want to make a booking, ask for:
- Name
- Date
- Time  
- Number of people

If they want to check/modify/cancel a booking, ask for their booking reference.

Response:"""
            
            response = self.llm.invoke(prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Fallback response error: {e}")
            return "I'm here to help with restaurant bookings. You can ask me to check availability, make a reservation, or manage existing bookings. What would you like to do?"
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session."""
        self.memory.clear()
        if session_id and session_id in self.session_data:
            del self.session_data[session_id]