"""LangChain agent implementation for restaurant bookings."""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
import logging
from booking_client import BookingAPIClient
from tools import (
    CheckAvailabilityTool, CreateBookingTool, GetBookingTool,
    UpdateBookingTool, CancelBookingTool
)

logger = logging.getLogger(__name__)


class BookingAgent:
    """Conversational agent for restaurant bookings."""
    
    def __init__(self, api_client: BookingAPIClient, llm_model: str = "gpt-4", temperature: float = 0.7):
        self.api_client = api_client
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
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
        self.agent_executor = self._create_agent()
        
        # Store session data
        self.session_data: Dict[str, Any] = {}
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with custom prompt."""
        
        prompt = PromptTemplate.from_template("""You are a helpful restaurant booking assistant for TheHungryUnicorn restaurant. 
        You help customers make, check, modify, and cancel their reservations.

        You have access to the following tools:
        {tools}

        When helping customers:
        1. Be friendly and professional
        2. Ask for missing information when needed (name, date, time, party size for new bookings)
        3. Confirm important details before making changes
        4. Provide booking references after creating reservations
        5. Handle dates naturally (understand "next Friday", "this weekend", etc.)

        Previous conversation:
        {chat_history}

        Customer: {input}
        
        Use this format:
        Thought: Consider what the customer needs
        Action: the action to take (one of [{tool_names}])
        Action Input: the input to the action
        Observation: the result of the action
        ... (repeat if necessary)
        Thought: I now know the final answer
        Final Answer: the final answer to the customer

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
            max_iterations=5
        )
    
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
            
            # Run the agent
            response = self.agent_executor.run(input=message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session."""
        self.memory.clear()
        if session_id and session_id in self.session_data:
            del self.session_data[session_id]