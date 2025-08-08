"""Enhanced LangChain agent implementation with OpenAI and Ollama support."""

from langchain.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from typing import List, Dict, Any, Optional, Union
import logging
import re
import os
from booking_client import BookingAPIClient
from tools import (
    CheckAvailabilityTool, CreateBookingTool, GetBookingTool,
    UpdateBookingTool, CancelBookingTool
)

logger = logging.getLogger(__name__)


class BookingAgent:
    """Conversational agent for restaurant bookings with multi-LLM support."""
    
    def __init__(
        self, 
        api_client: BookingAPIClient, 
        llm_provider: str = "openai",
        llm_model: str = None, 
        temperature: float = 0.3,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the booking agent.
        
        Args:
            api_client: BookingAPIClient instance
            llm_provider: "openai" or "ollama"
            llm_model: Model name (e.g., "gpt-4" for OpenAI, "llama2" for Ollama)
            temperature: LLM temperature setting
            ollama_base_url: Base URL for Ollama server
        """
        self.api_client = api_client
        self.llm_provider = llm_provider.lower()
        self.temperature = temperature
        
        # Initialize LLM based on provider
        if self.llm_provider == "ollama":
            if not llm_model:
                llm_model = "llama2"  # Default Ollama model
            self.llm = ChatOllama(
                model=llm_model,
                temperature=temperature,
                base_url=ollama_base_url
            )
            logger.info(f"Using Ollama with model: {llm_model}")
        else:  # Default to OpenAI
            if not llm_model:
                llm_model = "gpt-4"  # Default OpenAI model
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature
            )
            logger.info(f"Using OpenAI with model: {llm_model}")
        
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
        
        # Store session data for context
        self.session_data: Dict[str, Any] = {}
        self.context_tracker: Dict[str, Any] = {}
    
    def _create_agent(self) -> AgentExecutor:
        """Create an agent based on the LLM provider."""
        
        system_prompt = """You are a friendly and professional restaurant booking assistant for TheHungryUnicorn restaurant. 
Your goal is to help customers make, check, modify, and cancel their reservations efficiently.

IMPORTANT GUIDELINES:

1. **Collecting Information**:
   - For new bookings, you need: customer name, date, time, party size, and optionally contact number
   - Accept simple numeric inputs for party size (e.g., "4" means 4 people)
   - Parse natural language dates like "tomorrow", "next Friday", "this weekend"
   - Understand time formats like "7pm", "19:00", "7:30 PM"

2. **Customer Names**:
   - ALWAYS ask for the customer's actual name when making a booking
   - NEVER use tool names or action names as customer names
   - If the user hasn't provided their name, politely ask for it

3. **Response Formatting**:
   - Use proper markdown formatting in responses
   - Use emojis to make responses friendly
   - Include line breaks for readability
   - Format booking confirmations clearly with each detail on a new line

4. **Conversation Flow**:
   - Be conversational and natural
   - Confirm details before making changes
   - Provide booking references prominently after creating reservations
   - Handle errors gracefully with helpful suggestions

5. **Information Extraction**:
   - When user says just a number in response to "how many people", interpret it as party size
   - Be flexible with date/time formats
   - Remember context from previous messages

Remember: You're helping real customers, so be warm, helpful, and efficient."""

        if self.llm_provider == "ollama":
            # Use ReAct agent for Ollama (better compatibility)
            prompt = PromptTemplate.from_template(f"""{system_prompt}

You have access to the following tools:
{{tools}}

Previous conversation:
{{chat_history}}

Customer: {{input}}

Use this format:
Thought: Consider what the customer needs
Action: the action to take (one of [{{tool_names}}])
Action Input: the input to the action
Observation: the result of the action
... (repeat if necessary)
Thought: I now know the final answer
Final Answer: the final answer to the customer

{{agent_scratchpad}}""")
            
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            logger.info("Created ReAct agent for Ollama")
            
        else:
            # Use OpenAI Tools agent for OpenAI (better performance)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            logger.info("Created OpenAI Tools agent")
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False
        )
    
    def _extract_context(self, message: str, session_id: str) -> Dict[str, Any]:
        """Extract and maintain context from user messages."""
        context = self.context_tracker.get(session_id, {})
        
        # Extract party size from simple numbers
        if message.strip().isdigit():
            context['party_size'] = int(message.strip())
            return context
        
        # Extract party size from text
        party_patterns = [
            r'(\d+)\s*(?:people|persons?|guests?|pax)',
            r'(?:table\s+for|party\s+of)\s*(\d+)',
            r'(\d+)\s*(?:of\s+us)',
        ]
        for pattern in party_patterns:
            match = re.search(pattern, message.lower())
            if match:
                context['party_size'] = int(match.group(1))
                break
        
        # Extract time
        time_patterns = [
            r'(\d{1,2})\s*(?::(\d{2}))?\s*([ap]m)',
            r'(\d{1,2}):(\d{2})',
        ]
        for pattern in time_patterns:
            match = re.search(pattern, message.lower())
            if match:
                context['time'] = match.group(0)
                break
        
        # Extract date references
        date_keywords = ['today', 'tomorrow', 'weekend', 'monday', 'tuesday', 'wednesday', 
                        'thursday', 'friday', 'saturday', 'sunday']
        for keyword in date_keywords:
            if keyword in message.lower():
                context['date_reference'] = keyword
                break
        
        # Extract name (look for capitalized words that aren't common words)
        if 'name' in message.lower() or 'i am' in message.lower() or "i'm" in message.lower():
            # Try to extract name after "my name is", "I am", "I'm"
            name_patterns = [
                r"(?:my\s+name\s+is|i\s+am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$",  # Just a name on its own line
            ]
            for pattern in name_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    potential_name = match.group(1).strip()
                    # Filter out tool names and common words
                    if potential_name.lower() not in ['check', 'cancel', 'update', 'create', 'booking', 'reservation']:
                        context['customer_name'] = potential_name
                        break
        
        self.context_tracker[session_id] = context
        return context
    
    def _format_response(self, response: str) -> str:
        """Ensure proper formatting of the response."""
        # Fix markdown formatting
        response = response.replace('**', '**')
        response = response.replace('📋 **', '📋 **')
        response = response.replace('👤 **', '\n👤 **')
        response = response.replace('📅 **', '\n📅 **')
        response = response.replace('🕐 **', '\n🕐 **')
        response = response.replace('👥 **', '\n👥 **')
        
        # Ensure line breaks are preserved
        response = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', response)
        
        return response
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """
        Process a user message and return the agent's response.
        
        Args:
            message: The user's message
            session_id: Optional session ID for maintaining conversation state
        
        Returns:
            The agent's response with proper formatting
        """
        try:
            # Initialize session if needed
            if session_id:
                if session_id not in self.session_data:
                    self.session_data[session_id] = {}
                    self.context_tracker[session_id] = {}
                
                # Extract context from message
                context = self._extract_context(message, session_id)
                
                # Store context in session
                self.session_data[session_id].update(context)
                
                # Augment message with context if it's a simple number
                if message.strip().isdigit():
                    message = f"{message} people"
            
            # Run the agent
            response = self.agent_executor.invoke({"input": message})
            
            # Extract the actual response
            if isinstance(response, dict):
                response_text = response.get('output', str(response))
            else:
                response_text = str(response)
            
            # Format the response
            formatted_response = self._format_response(response_text)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = "I apologize, but I encountered an error processing your request."
            
            # Add provider-specific error messages
            if self.llm_provider == "ollama":
                error_msg += " Please ensure the Ollama server is running (ollama serve)."
            elif "api" in str(e).lower() and "key" in str(e).lower():
                error_msg += " Please check your OpenAI API key configuration."
            
            return error_msg
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session."""
        self.memory.clear()
        if session_id:
            if session_id in self.session_data:
                del self.session_data[session_id]
            if session_id in self.context_tracker:
                del self.context_tracker[session_id]