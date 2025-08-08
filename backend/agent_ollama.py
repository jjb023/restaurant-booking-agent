"""LangChain agent implementation using Ollama for restaurant bookings."""

import os
import logging
from typing import Optional, Dict, Any
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from booking_client import BookingAPIClient
from tools import CheckAvailabilityTool, CreateBookingTool, GetBookingTool, UpdateBookingTool, CancelBookingTool

logger = logging.getLogger(__name__)

class BookingAgent:
    def __init__(self, api_client: BookingAPIClient, model_name: str = "llama3.2:3b", 
                 temperature: float = 0.3, base_url: str = "http://localhost:11434"):
        self.api_client = api_client
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )
        
        # Initialize tools
        self.tools = {
            'check_availability': CheckAvailabilityTool(api_client),
            'create_booking': CreateBookingTool(api_client),
            'get_booking': GetBookingTool(api_client),
            'update_booking': UpdateBookingTool(api_client),
            'cancel_booking': CancelBookingTool(api_client)
        }
        
        # Session data for conversation state
        self.session_data = {}
        
        # System prompt for natural language understanding
        self.system_prompt = """You are a helpful restaurant booking assistant for TheHungryUnicorn restaurant. 

Your job is to understand what the user wants and respond naturally. You can:

1. **Greet users warmly** - Be friendly and welcoming
2. **Understand booking requests** - Extract name, date, time, party size
3. **Handle availability checks** - Understand when users want to check available times
4. **Manage existing bookings** - Help with viewing, updating, or canceling bookings
5. **Provide helpful responses** - Give clear, friendly answers with emojis when appropriate

**Important Guidelines:**
- Be polite, professional, and enthusiastic
- Use emojis to make responses friendly
- Ask for missing information when needed
- Confirm details before taking actions
- Provide clear next steps

**Available Actions:**
- check_availability(date, time, party_size)
- create_booking(name, date, time, party_size)
- get_booking(reference)
- update_booking(reference, details)
- cancel_booking(reference)

Respond naturally and helpfully!"""

    def _extract_info(self, message: str) -> Dict[str, Any]:
        """Extract structured information from user message using LLM."""
        try:
            # Use LLM to extract information with a more explicit prompt
            extraction_prompt = f"""Extract booking information from: "{message}"

IMPORTANT: Only extract information that is EXPLICITLY mentioned in the message.
Do NOT assume or guess any values. Use null for anything not clearly stated.

Return ONLY a JSON object with these fields (use null if not found):
- intent: "greeting", "booking", "availability", "check_booking", "cancel_booking", "update_booking"
- name: customer name (only if explicitly mentioned)
- date: date (e.g., "tomorrow", "saturday", "2025-01-15")
- time: time (e.g., "7pm", "19:30")
- party_size: number of people (only if explicitly mentioned)
- reference: booking reference number

Example: {{"intent": "booking", "name": null, "date": "tomorrow", "time": "7pm", "party_size": null, "reference": null}}

JSON response:"""

            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            content = response.content.strip()
            
            # Try to parse JSON from the response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    extracted_info = json.loads(json_match.group())
                    logger.info(f"Extracted info: {extracted_info}")
                    return extracted_info
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from: {content}")
            
            # If LLM didn't return proper JSON, use fallback pattern matching
            logger.info("LLM didn't return proper JSON, using fallback pattern matching")
            return self._fallback_extract_info(message)
            
        except Exception as e:
            logger.error(f"Error extracting info: {e}")
            return self._fallback_extract_info(message)

    def _fallback_extract_info(self, message: str) -> Dict[str, Any]:
        """Fallback pattern matching when LLM extraction fails."""
        message_lower = message.lower().strip()
        
        # Check for greetings
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return {"intent": "greeting", "name": None, "date": None, "time": None, "party_size": None, "reference": None}
        
        # Check for availability requests
        if any(word in message_lower for word in ['availability', 'available', 'check', 'times']) and any(word in message_lower for word in ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'tomorrow', 'today', 'weekend']):
            # Extract date from message
            date = self._extract_date_from_message(message)
            return {"intent": "availability", "name": None, "date": date, "time": None, "party_size": None, "reference": None}
        
        # Check for booking requests
        if any(word in message_lower for word in ['book', 'reserve', 'table', 'reservation']):
            return {"intent": "booking", "name": None, "date": None, "time": None, "party_size": None, "reference": None}
        
        # Check for check booking requests
        if any(word in message_lower for word in ['check my booking', 'my booking', 'my reservation', 'view booking', 'show booking']):
            return {"intent": "check_booking", "name": None, "date": None, "time": None, "party_size": None, "reference": None}
        
        # Check for cancel booking requests
        if any(word in message_lower for word in ['cancel my reservation', 'cancel booking', 'cancel reservation']):
            return {"intent": "cancel_booking", "name": None, "date": None, "time": None, "party_size": None, "reference": None}
        
        # Check if message is just a number (likely a booking reference)
        if message.strip().isdigit():
            return {"intent": "check_booking", "name": None, "date": None, "time": None, "party_size": None, "reference": message.strip()}
        
        # Check for complete booking info (comma-separated) - be more conservative
        if ',' in message:
            parts = [part.strip() for part in message.split(',')]
            if len(parts) >= 4:
                # Try to extract booking info from comma-separated format
                name = parts[0].strip()
                # Only use name if it looks like a real name (not a date/time word)
                if not any(word in name.lower() for word in ['tomorrow', 'today', 'saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'pm', 'am', 'people']):
                    date = self._extract_date_from_message(parts[1].strip())
                    time = self._extract_time_from_message(parts[2].strip())
                    party_size = self._extract_party_size_from_message(parts[3].strip())
                    
                    if all([name, date, time, party_size]):
                        return {"intent": "booking", "name": name, "date": date, "time": time, "party_size": party_size, "reference": None}
        
        # Default to unknown
        return {"intent": "unknown", "name": None, "date": None, "time": None, "party_size": None, "reference": None}

    def _generate_response(self, message: str, context: str = "") -> str:
        """Generate a natural response using the LLM."""
        try:
            prompt = f"""Context: {context}

User message: "{message}"

Respond naturally and helpfully as a restaurant booking assistant. Be friendly and use emojis when appropriate."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble understanding. Could you please try again?"

    def _check_availability(self, date: str, time: Optional[str] = None, party_size: Optional[int] = None) -> str:
        """Check availability using the tool."""
        try:
            result = self.tools['check_availability']._run(date, time, party_size)
            return result
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return f"Sorry, I couldn't check availability. Error: {str(e)}"

    def _create_booking(self, name: str, date: str, time: str, party_size: int) -> str:
        """Create a booking using the tool."""
        try:
            # Convert natural language date to YYYY-MM-DD format
            converted_date = self._convert_natural_date(date)
            if converted_date:
                date = converted_date
            
            # Convert natural language time to 24-hour format
            converted_time = self._convert_natural_time(time)
            if converted_time:
                time = converted_time
            
            result = self.tools['create_booking']._run(name, date, time, party_size)
            return result
        except Exception as e:
            logger.error(f"Error creating booking: {e}")
            return f"Sorry, I couldn't create the booking. Error: {str(e)}"

    def _get_booking(self, reference: str) -> str:
        """Get booking details using the tool."""
        try:
            result = self.tools['get_booking']._run(reference)
            return result
        except Exception as e:
            logger.error(f"Error getting booking: {e}")
            return f"Sorry, I couldn't find that booking. Error: {str(e)}"

    def _cancel_booking(self, reference: str) -> str:
        """Cancel a booking using the tool."""
        try:
            result = self.tools['cancel_booking']._run(reference)
            return result
        except Exception as e:
            logger.error(f"Error canceling booking: {e}")
            return f"Sorry, I couldn't cancel that booking. Error: {str(e)}"

    def _extract_date_from_message(self, message: str) -> Optional[str]:
        """Extract date from message."""
        import re
        message_lower = message.lower()
        
        # Check for date patterns
        date_patterns = [
            r'\b(tomorrow|today)\b',
            r'\b(saturday|sunday|monday|tuesday|wednesday|thursday|friday)\b',
            r'\b(next\s+(saturday|sunday|monday|tuesday|wednesday|thursday|friday))\b',
            r'\b(this\s+(saturday|sunday|monday|tuesday|wednesday|thursday|friday))\b',
            r'\b(weekend)\b',
            r'\b(\d{4}-\d{2}-\d{2})\b'  # YYYY-MM-DD format
        ]
        
        for pattern in date_patterns:
            if match := re.search(pattern, message_lower):
                return match.group(1) if match.groups() else match.group(0)
        
        return None

    def _extract_time_from_message(self, message: str) -> Optional[str]:
        """Extract time from message."""
        import re
        message_lower = message.lower()
        
        # Check for time patterns
        time_patterns = [
            r'\b(\d{1,2}:\d{2})\b',  # HH:MM format
            r'\b(\d{1,2}(?::\d{2})?\s*(am|pm))\b',  # 7pm, 7:30pm format
            r'\b(\d{1,2})\s*(am|pm)\b'  # 7 am format
        ]
        
        for pattern in time_patterns:
            if match := re.search(pattern, message_lower):
                return match.group(0)
        
        return None

    def _extract_party_size_from_message(self, message: str) -> Optional[int]:
        """Extract party size from message."""
        import re
        message_lower = message.lower()
        
        # Check for party size patterns
        party_patterns = [
            r'\b(\d+)\s*(?:people?|guests?|persons?)\b',
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
        
        # Also check for just numbers in the message
        number_match = re.search(r'\b(\d+)\b', message_lower)
        if number_match:
            try:
                num = int(number_match.group(1))
                if 1 <= num <= 20:
                    return num
            except ValueError:
                pass
        
        return None

    def _convert_natural_date(self, date_str: str) -> Optional[str]:
        """Convert natural language date to YYYY-MM-DD format using dateparser."""
        try:
            import dateparser
            from datetime import datetime
            
            # Parse the natural language date
            parsed_date = dateparser.parse(date_str)
            if parsed_date:
                # Return in YYYY-MM-DD format
                return parsed_date.strftime('%Y-%m-%d')
            return None
        except ImportError:
            logger.warning("dateparser not installed, using original date string")
            return date_str
        except Exception as e:
            logger.warning(f"Could not parse date '{date_str}': {e}")
            return date_str

    def _convert_natural_time(self, time_str: str) -> Optional[str]:
        """Convert natural language time to 24-hour format."""
        try:
            import re
            from datetime import datetime
            
            time_lower = time_str.lower().strip()
            
            # Handle common time formats
            if 'pm' in time_lower:
                # Extract hour
                hour_match = re.search(r'(\d{1,2})', time_lower)
                if hour_match:
                    hour = int(hour_match.group(1))
                    if hour != 12:
                        hour += 12
                    return f"{hour:02d}:00"
            elif 'am' in time_lower:
                # Extract hour
                hour_match = re.search(r'(\d{1,2})', time_lower)
                if hour_match:
                    hour = int(hour_match.group(1))
                    if hour == 12:
                        hour = 0
                    return f"{hour:02d}:00"
            elif ':' in time_str:
                # Already in HH:MM format
                return time_str
            
            return None
        except Exception as e:
            logger.warning(f"Could not parse time '{time_str}': {e}")
            return None

    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """Process a user message and return a response."""
        try:
            # Initialize session if needed
            if session_id and session_id not in self.session_data:
                self.session_data[session_id] = {
                    'booking_info': {},
                    'conversation_step': 'greeting'
                }
            
            session = self.session_data[session_id] if session_id else {}
            booking_info = session.get('booking_info', {})
            conversation_step = session.get('conversation_step', 'greeting')
            
            # Extract information from the message
            extracted_info = self._extract_info(message)
            intent = extracted_info.get('intent', 'unknown')
            
            logger.info(f"Extracted intent: {intent}, info: {extracted_info}")
            
            # Handle different intents with proper routing
            if intent == 'greeting':
                return """🍽️ Welcome to TheHungryUnicorn! I can help you with:

• 📅 Check availability for any date
• 🎯 Make a new reservation
• 📋 View your existing booking details
• ✏️ Modify or cancel your reservation

What would you like to do today?"""
            
            elif intent == 'availability':
                date = extracted_info.get('date')
                if date:
                    return self._check_availability(date)
                else:
                    return """📅 I can check availability for you! 

What date and time are you interested in? You can say things like:
• "tomorrow"
• "saturday"
• "next friday"
• "2025-08-15"

What date would you like to check?"""
            
            elif intent == 'check_booking':
                reference = extracted_info.get('reference')
                if reference:
                    return self._get_booking(reference)
                else:
                    return """📋 I can help you check your booking details! 

Please provide your booking reference number (it's usually a 6-8 character code like "ABC123" or just a number like "5")."""
            
            elif intent == 'cancel_booking':
                reference = extracted_info.get('reference')
                if reference:
                    return self._cancel_booking(reference)
                else:
                    return """❌ I can help you cancel your reservation! 

Please provide your booking reference number to cancel your reservation."""
            
            elif intent == 'booking':
                # Update session with any new information
                if session_id:
                    if extracted_info.get('name'):
                        session['booking_info']['name'] = extracted_info['name']
                    if extracted_info.get('date'):
                        session['booking_info']['date'] = extracted_info['date']
                    if extracted_info.get('time'):
                        session['booking_info']['time'] = extracted_info['time']
                    if extracted_info.get('party_size'):
                        session['booking_info']['party_size'] = extracted_info['party_size']
                
                # Check what we have and what we need
                current_info = session.get('booking_info', {})
                name = current_info.get('name') or extracted_info.get('name')
                date = current_info.get('date') or extracted_info.get('date')
                time = current_info.get('time') or extracted_info.get('time')
                party_size = current_info.get('party_size') or extracted_info.get('party_size')
                
                # If we have all the information, create the booking
                if all([name, date, time, party_size]):
                    if session_id:
                        # Clear the session after successful booking
                        session['booking_info'] = {}
                        session['conversation_step'] = 'greeting'
                    return self._create_booking(name, date, time, party_size)
                else:
                    # Determine what's missing and ask for it
                    missing_fields = []
                    if not name:
                        missing_fields.append("name")
                    if not date:
                        missing_fields.append("date")
                    if not time:
                        missing_fields.append("time")
                    if not party_size:
                        missing_fields.append("party_size")
                    
                    # Provide context about what we already have
                    context_parts = []
                    if name:
                        context_parts.append(f"Name: {name}")
                    if date:
                        context_parts.append(f"Date: {date}")
                    if time:
                        context_parts.append(f"Time: {time}")
                    if party_size:
                        context_parts.append(f"Party size: {party_size}")
                    
                    context = ", ".join(context_parts) if context_parts else "No details yet"
                    
                    # Ask for the next missing field
                    if not name:
                        if session_id:
                            session['conversation_step'] = 'asking_name'
                        return f"""🎯 I'd be happy to help you make a reservation! 

Current details: {context}

What's your name?"""
                    elif not date:
                        if session_id:
                            session['conversation_step'] = 'asking_date'
                        return f"""Thanks {name}! 

Current details: {context}

What date would you like to visit? (e.g., 'tomorrow', 'saturday', 'next friday')"""
                    elif not time:
                        if session_id:
                            session['conversation_step'] = 'asking_time'
                        return f"""Great! 

Current details: {context}

What time would you prefer? (e.g., '7pm', '19:30')"""
                    elif not party_size:
                        if session_id:
                            session['conversation_step'] = 'asking_party_size'
                        return f"""Perfect! 

Current details: {context}

How many people will be in your party?"""
            
            # Handle step-by-step conversation with better context
            elif conversation_step == 'asking_name':
                name = extracted_info.get('name') or message.strip()
                if session_id:
                    session['booking_info']['name'] = name
                    session['conversation_step'] = 'asking_date'
                return f"Thanks {name}! What date would you like to visit? (e.g., 'tomorrow', 'saturday', 'next friday')"
            
            elif conversation_step == 'asking_date':
                date = extracted_info.get('date') or message.strip()
                if session_id:
                    session['booking_info']['date'] = date
                    session['conversation_step'] = 'asking_time'
                return "What time would you prefer? (e.g., '7pm', '19:30')"
            
            elif conversation_step == 'asking_time':
                time = extracted_info.get('time') or message.strip()
                if session_id:
                    session['booking_info']['time'] = time
                    session['conversation_step'] = 'asking_party_size'
                return "How many people will be in your party?"
            
            elif conversation_step == 'asking_party_size':
                party_size = extracted_info.get('party_size')
                if party_size and session_id:
                    session['booking_info']['party_size'] = party_size
                    # Create the booking with all collected info
                    booking_info = session['booking_info']
                    # Clear session after successful booking
                    session['booking_info'] = {}
                    session['conversation_step'] = 'greeting'
                    return self._create_booking(
                        booking_info['name'], 
                        booking_info['date'], 
                        booking_info['time'], 
                        booking_info['party_size']
                    )
            
            # Handle unknown intent or fallback
            else:
                # Check if this might be a booking reference number
                if message.strip().isdigit():
                    return self._get_booking(message.strip())
                else:
                    # Fallback: generate a natural response
                    return self._generate_response(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm having trouble understanding. Could you please try again?"