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
        
        # Initialize Ollama LLM - simpler configuration
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url
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
        
        # ReAct prompt with all required variables
        prompt = PromptTemplate.from_template("""You are a helpful restaurant booking assistant for TheHungryUnicorn restaurant.

You have access to the following tools:

{tools}

Use a tool to answer questions or perform actions. You can use these tools: {tool_names}

When helping customers:
1. Be friendly and professional
2. Ask for missing information when needed
3. Confirm important details before making changes
4. Provide booking references after creating reservations

Previous conversation:
{chat_history}

Question: {input}

To answer, use this exact format:

Thought: I need to figure out what the customer wants
Action: [one of {tool_names}]
Action Input: the input parameters for the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to be helpful and friendly.

{agent_scratchpad}""")
        
        try:
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
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """Process a user message and return the agent's response."""
        try:
            # Store any extracted booking info in session
            if session_id and session_id not in self.session_data:
                self.session_data[session_id] = {}
            
            # Run the agent with invoke
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
            # Parse message for intent
            message_lower = message.lower()
            
            # Check for common intents
            if any(word in message_lower for word in ['book', 'reserve', 'table']):
                return "I'd be happy to help you make a reservation! I'll need a few details:\n- Your name\n- Date you'd like to visit\n- Time you prefer\n- Number of people in your party"
            
            elif any(word in message_lower for word in ['availability', 'available', 'free']):
                return "I can check availability for you! What date and time are you interested in?"
            
            elif any(word in message_lower for word in ['cancel', 'cancellation']):
                return "I can help you cancel your reservation. Could you please provide your booking reference?"
            
            elif any(word in message_lower for word in ['change', 'modify', 'update']):
                return "I can help you modify your reservation. Please provide your booking reference and what you'd like to change."
            
            elif any(word in message_lower for word in ['check', 'status', 'my booking', 'my reservation']):
                return "I can look up your reservation details. Please provide your booking reference."
            
            else:
                # General response using LLM
                prompt = f"""You are a helpful restaurant booking assistant for TheHungryUnicorn.
                
Customer message: {message}

Provide a brief, helpful response. If they want to make a booking, ask for: name, date, time, and party size.
If they want to check/modify/cancel, ask for their booking reference.

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
            return "I'm here to help with restaurant bookings at TheHungryUnicorn. You can:\n• Make a new reservation\n• Check availability\n• View your booking details\n• Modify or cancel existing bookings\n\nWhat would you like to do?"
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session."""
        self.memory.clear()
        if session_id and session_id in self.session_data:
            del self.session_data[session_id]