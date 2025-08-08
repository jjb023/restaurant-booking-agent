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
        
        # Initialize Ollama LLM with optimized settings
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            timeout=30
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
        
        # Create the agent with enhanced prompt for better conversation flow
        self.agent_executor = self._create_agent()
        
        # Store session data for better context management
        self.session_data: Dict[str, Any] = {}
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with enhanced prompt for better conversation flow."""
        
        # Enhanced prompt with better conversation guidelines and clearer ReAct format
        prompt = PromptTemplate.from_template("""You are a helpful and friendly restaurant booking assistant for TheHungryUnicorn restaurant.

IMPORTANT GUIDELINES:
1. Always be polite, professional, and enthusiastic
2. When you understand booking details from the user, CONFIRM them before taking action
3. Use the tools to perform actual booking operations
4. If the user provides booking details (name, date, time, party size), confirm them and then use create_booking tool
5. If the user asks for availability, use check_availability tool with a specific date
6. Always provide clear, helpful responses with emojis when appropriate
7. Remember context from previous messages in the conversation
8. DO NOT call tools with None or empty parameters - only call tools when you have valid information
9. For simple greetings like "hello", "hi", "hey" - just provide a friendly welcome message WITHOUT using any tools
10. Only use tools when the user specifically asks for booking operations or availability checks
11. After using a tool, provide a clear final answer summarizing the result

Available tools: {tool_names}

You have access to these tools:

{tools}

Previous conversation:
{chat_history}

Current question: {input}

To answer, use this exact format:

Thought: I need to figure out what the customer wants
Action: [one of {tool_names}]
Action Input: the input parameters for the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to be helpful, friendly, and provide excellent customer service.

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
                max_iterations=2,  # Reduced from 3 to 2
                early_stopping_method="force"  # Changed from "generate" to "force"
            )
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """Process a user message and return the agent's response."""
        try:
            # Store session data for better context management
            if session_id and session_id not in self.session_data:
                self.session_data[session_id] = {
                    'booking_info': {},
                    'conversation_step': 'greeting'
                }
            
            session = self.session_data[session_id] if session_id else {}
            booking_info = session.get('booking_info', {})
            conversation_step = session.get('conversation_step', 'greeting')
            
            # Check for simple greetings first - handle these directly without agent
            message_lower = message.lower().strip()
            if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
                if session_id:
                    session['conversation_step'] = 'greeting'
                return """🍽️ Welcome to TheHungryUnicorn! I can help you with:

• 📅 Check availability for any date
• 🎯 Make a new reservation
• 📋 View your existing booking details
• ✏️ Modify or cancel your reservation

What would you like to do today?"""
            
            # Handle availability requests directly
            if any(word in message_lower for word in ['availability', 'available', 'check', 'times']) and any(word in message_lower for word in ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'tomorrow', 'today', 'weekend']):
                # Extract date from message
                date = self._extract_date_from_message(message)
                if date:
                    return self._check_availability_direct(date)
                else:
                    return """📅 I can check availability for you! 

What date and time are you interested in? You can say things like:
• "tomorrow"
• "saturday"
• "next friday"
• "2025-08-15"

What date would you like to check?"""
            
            # Handle booking requests directly
            if any(word in message_lower for word in ['book', 'reserve', 'table', 'reservation']):
                booking_info = self._extract_booking_info_from_message(message)
                if booking_info and all(booking_info.values()):
                    return self._create_booking_direct(booking_info)
                else:
                    if session_id:
                        session['conversation_step'] = 'asking_name'
                    return """🎯 I'd be happy to help you make a reservation! 

To get started, I'll need a few details:
• Your name
• Date you'd like to visit (e.g., "tomorrow", "saturday", "2025-08-15")
• Time you prefer (e.g., "7pm", "19:30")
• Number of people in your party

What's your name?"""
            
            # Handle conversation flow based on current step
            if conversation_step == 'asking_name' and self._is_likely_name_input(message):
                # Extract name and store it
                name = self._extract_name_from_message(message)
                if name and session_id:
                    session['booking_info']['name'] = name
                    session['conversation_step'] = 'asking_date'
                return f"Thanks {name}! What date would you like to visit? (e.g., 'tomorrow', 'saturday', 'next friday')"
            
            elif conversation_step == 'asking_date' and self._is_likely_date_input(message):
                # Extract date and store it
                date = self._extract_date_from_message(message)
                if date and session_id:
                    session['booking_info']['date'] = date
                    session['conversation_step'] = 'asking_time'
                return "What time would you prefer? (e.g., '7pm', '19:30')"
            
            elif conversation_step == 'asking_time' and self._is_likely_time_input(message):
                # Extract time and store it
                time = self._extract_time_from_message(message)
                if time and session_id:
                    session['booking_info']['time'] = time
                    session['conversation_step'] = 'asking_party_size'
                return "How many people will be in your party?"
            
            elif conversation_step == 'asking_party_size' and self._is_likely_party_size_input(message):
                # Extract party size and create booking
                party_size = self._extract_party_size_from_message(message)
                if party_size and session_id:
                    session['booking_info']['party_size'] = party_size
                    # Create the booking with all collected info
                    return self._create_booking_direct(session['booking_info'])
            
            # For other cases, try the agent but with better error handling
            try:
                response = self.agent_executor.invoke({"input": message})
                
                # Debug: Log the response structure
                logger.info(f"Agent response type: {type(response)}")
                if isinstance(response, dict):
                    logger.info(f"Agent response keys: {response.keys()}")
                
                # Extract the output - handle different response formats
                if isinstance(response, dict):
                    if 'output' in response:
                        final_response = response['output']
                    elif 'result' in response:
                        final_response = response['result']
                    else:
                        # If no clear output, try to extract from the response
                        final_response = str(response)
                else:
                    final_response = str(response)
                
                # Debug: Log the final response
                logger.info(f"Final response: {final_response[:100]}...")
                
                return final_response
                
            except Exception as agent_error:
                logger.error(f"Agent error: {agent_error}")
                # Fallback to direct LLM response
                return self._fallback_response(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Fallback to direct response if agent fails
            return self._fallback_response(message)
    
    def _fallback_response(self, message: str) -> str:
        """Enhanced fallback response with better conversation handling."""
        try:
            # Parse message for intent
            message_lower = message.lower()
            
            # Check for simple greetings
            if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
                return """🍽️ Welcome to TheHungryUnicorn! I can help you with:

• 📅 Check availability for any date
• 🎯 Make a new reservation
• 📋 View your existing booking details
• ✏️ Modify or cancel your reservation

What would you like to do today?"""
            
            # Check for common intents with better responses
            if any(word in message_lower for word in ['book', 'reserve', 'table', 'reservation']):
                return """🎯 I'd be happy to help you make a reservation! 

To get started, I'll need a few details:
• Your name
• Date you'd like to visit (e.g., "tomorrow", "saturday", "2025-08-15")
• Time you prefer (e.g., "7pm", "19:30")
• Number of people in your party

What's your name?"""
            
            elif any(word in message_lower for word in ['availability', 'available', 'free', 'times']):
                return """📅 I can check availability for you! 

What date and time are you interested in? You can say things like:
• "tomorrow"
• "saturday"
• "next friday"
• "2025-08-15"

What date would you like to check?"""
            
            elif any(word in message_lower for word in ['cancel', 'cancellation']):
                return """❌ I can help you cancel your reservation. 

Could you please provide your booking reference? It's usually a 6-8 character code like "ABC123"."""
            
            elif any(word in message_lower for word in ['change', 'modify', 'update']):
                return """✏️ I can help you modify your reservation. 

Please provide:
• Your booking reference
• What you'd like to change (date, time, or party size)"""
            
            elif any(word in message_lower for word in ['check', 'status', 'my booking', 'my reservation']):
                return """📋 I can look up your reservation details. 

Please provide your booking reference to see your reservation details."""
            
            else:
                # General response using LLM with better prompt
                prompt = f"""You are a helpful restaurant booking assistant for TheHungryUnicorn restaurant.

Customer message: {message}

Provide a brief, friendly response. If they want to make a booking, ask for: name, date, time, and party size.
If they want to check/modify/cancel, ask for their booking reference.
Use emojis to make it friendly and engaging.

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
            return """🍽️ Welcome to TheHungryUnicorn! I can help you with:

• 📅 Check availability for any date
• 🎯 Make a new reservation
• 📋 View your existing booking details
• ✏️ Modify or cancel your reservation

What would you like to do today?"""
    
    def _extract_date_from_message(self, message: str) -> str:
        """Extract date from message using simple patterns."""
        message_lower = message.lower()
        from datetime import datetime, timedelta
        
        today = datetime.now()
        
        if "today" in message_lower:
            return today.strftime('%Y-%m-%d')
        elif "tomorrow" in message_lower:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif "weekend" in message_lower:
            days_ahead = 5 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        else:
            weekdays = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
            }
            for day, idx in weekdays.items():
                if day in message_lower:
                    days_ahead = (idx - today.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        return None
    
    def _extract_booking_info_from_message(self, message: str) -> dict:
        """Extract booking information from message."""
        import re
        info = {}
        message_lower = message.lower()
        
        # Extract name
        name_patterns = [
            r"(?:my name is|i am|i'm|it's|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$",
            r"(?:for|under the name of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in name_patterns:
            if match := re.search(pattern, message.strip(), re.I):
                name = match.group(1).strip()
                if not any(word in name.lower() for word in ['today', 'tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                    info['name'] = name
                    break
        
        # Extract date
        info['date'] = self._extract_date_from_message(message)
        
        # Extract time
        time_patterns = [
            r'(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>[ap]m)?',
            r'(?P<hour>\d{1,2})\s*(?P<ampm>[ap]m)',
            r'\b(?P<hour>\d{1,2})\b(?!\s*(?:people?|guests?|persons?))'
        ]
        
        for pattern in time_patterns:
            if match := re.search(pattern, message_lower):
                try:
                    hour = int(match.group("hour"))
                    minute = int(match.group("minute")) if "minute" in match.groupdict() and match.group("minute") else 0
                    meridiem = match.group("ampm") if "ampm" in match.groupdict() else None

                    if meridiem == 'pm' and hour < 12:
                        hour += 12
                    elif meridiem == 'am' and hour == 12:
                        hour = 0

                    if 10 <= hour <= 23:
                        info['time'] = f"{hour:02d}:{minute:02d}:00"
                        break
                except ValueError:
                    continue
        
        # Extract party size
        party_patterns = [
            r'\b(\d+)\s*(?:people?|guests?|persons?)\b',
            r'\b(?:party of|table for|reservation for)\s*(\d+)\b',
            r'\b(\d+)\b(?!\s*(?:pm|am|:))'
        ]
        
        for pattern in party_patterns:
            if match := re.search(pattern, message_lower):
                num = int(match.group(1))
                if 1 <= num <= 20:
                    info['party_size'] = num
                    break
        
        return info
    
    def _check_availability_direct(self, date: str) -> str:
        """Check availability directly without agent."""
        try:
            result = self.api_client.check_availability(date)
            
            if result['success']:
                slots = result.get('data', {}).get('available_slots', [])
                if slots:
                    times = [s['time'].replace(':00', '') if isinstance(s, dict) else str(s).replace(':00', '') 
                            for s in slots[:8]]
                    return f"✅ Available times for {date}:\n" + \
                           "\n".join(f"• {t}" for t in times) + \
                           f"\n\n🎯 To book one of these times, just reply with your preferred time!"
                else:
                    return f"❌ No available slots for {date}. Would you like to try another date?"
            return "❌ Couldn't check availability. Please try again."
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return "❌ Error checking availability. Please try again."
    
    def _create_booking_direct(self, info: dict) -> str:
        """Create booking directly without agent."""
        try:
            result = self.api_client.create_booking(
                customer_name=info['name'],
                date=info['date'],
                time=info['time'],
                party_size=info['party_size']
            )
            
            if result['success']:
                booking_id = result['data'].get('booking_id')
                time_display = info['time'].replace(':00', '')
                
                return f"""🎉 Perfect! Your reservation is confirmed!

📋 **Booking Reference:** {booking_id}
👤 **Name:** {info['name']}
📅 **Date:** {info['date']}
🕐 **Time:** {time_display}
👥 **Party:** {info['party_size']} people

💡 **Important:** Please save your booking reference ({booking_id}) for future reference.

See you soon at TheHungryUnicorn! 🍽️"""
            else:
                return f"❌ Couldn't create booking: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Error creating booking: {e}")
            return "❌ Error creating booking. Please try again."
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session."""
        self.memory.clear()
        if session_id and session_id in self.session_data:
            del self.session_data[session_id]
    
    def _is_likely_name_input(self, message: str) -> bool:
        """Check if message is likely a name input."""
        import re
        message = message.strip()
        
        # Check if it looks like a name (first and last name, capitalized)
        name_pattern = r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'
        if re.match(name_pattern, message):
            return True
        
        # Check for common name indicators
        name_indicators = ['my name is', 'i am', "i'm", 'call me', 'this is']
        message_lower = message.lower()
        if any(indicator in message_lower for indicator in name_indicators):
            return True
        
        return False
    
    def _is_likely_date_input(self, message: str) -> bool:
        """Check if message is likely a date input."""
        message_lower = message.lower()
        
        # Check for date keywords
        date_keywords = ['tomorrow', 'today', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'weekend']
        if any(keyword in message_lower for keyword in date_keywords):
            return True
        
        # Check for date patterns
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
        ]
        for pattern in date_patterns:
            if re.search(pattern, message):
                return True
        
        return False
    
    def _is_likely_time_input(self, message: str) -> bool:
        """Check if message is likely a time input."""
        import re
        message_lower = message.lower()
        
        # Check for time patterns
        time_patterns = [
            r'\d{1,2}:\d{2}',  # HH:MM
            r'\d{1,2}\s*[ap]m',  # 7pm, 7 pm
            r'\d{1,2}$',  # Just a number (assume time)
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, message_lower):
                return True
        
        return False
    
    def _is_likely_party_size_input(self, message: str) -> bool:
        """Check if message is likely a party size input."""
        import re
        message_lower = message.lower()
        
        # Check for party size patterns
        party_patterns = [
            r'\d+\s*(?:people?|guests?|persons?)',
            r'party of\s*\d+',
            r'for\s*\d+',
            r'^\d+$',  # Just a number
        ]
        
        for pattern in party_patterns:
            if re.search(pattern, message_lower):
                return True
        
        return False
    
    def _extract_name_from_message(self, message: str) -> str:
        """Extract name from message."""
        import re
        message = message.strip()
        
        # Check if it looks like a name (first and last name, capitalized)
        name_pattern = r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'
        if match := re.match(name_pattern, message):
            return match.group(0)
        
        # Check for common name indicators
        name_indicators = ['my name is', 'i am', "i'm", 'call me', 'this is']
        message_lower = message.lower()
        for indicator in name_indicators:
            if indicator in message_lower:
                # Extract the name after the indicator
                name_part = message_lower.split(indicator)[-1].strip()
                if name_part:
                    return name_part.title()
        
        return message.title()  # Default to the message as name
    
    def _extract_time_from_message(self, message: str) -> str:
        """Extract time from message."""
        import re
        message_lower = message.lower()
        
        # Check for time patterns
        time_patterns = [
            r'(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>[ap]m)?',
            r'(?P<hour>\d{1,2})\s*(?P<ampm>[ap]m)',
            r'\b(?P<hour>\d{1,2})\b(?!\s*(?:people?|guests?|persons?))'
        ]
        
        for pattern in time_patterns:
            if match := re.search(pattern, message_lower):
                try:
                    hour = int(match.group("hour"))
                    minute = int(match.group("minute")) if "minute" in match.groupdict() and match.group("minute") else 0
                    meridiem = match.group("ampm") if "ampm" in match.groupdict() else None

                    if meridiem == 'pm' and hour < 12:
                        hour += 12
                    elif meridiem == 'am' and hour == 12:
                        hour = 0

                    if 10 <= hour <= 23:
                        return f"{hour:02d}:{minute:02d}:00"
                except ValueError:
                    continue
        
        return None
    
    def _extract_party_size_from_message(self, message: str) -> int:
        """Extract party size from message."""
        import re
        message_lower = message.lower()
        
        # Check for party size patterns
        party_patterns = [
            r'\b(\d+)\s*(?:people?|guests?|persons?)',
            r'party of\s*(\d+)',
            r'for\s*(\d+)',
            r'^(\d+)$'  # Just a number
        ]
        
        for pattern in party_patterns:
            if match := re.search(pattern, message_lower):
                try:
                    num = int(match.group(1))
                    if 1 <= num <= 20:
                        return num
                except ValueError:
                    continue
        
        return None