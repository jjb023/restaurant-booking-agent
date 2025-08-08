"""Simple Ollama-based agent for restaurant bookings."""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from booking_client import BookingAPIClient
from tools import DateTimeParser

logger = logging.getLogger(__name__)


class SimpleBookingAgent:
    """Simple conversational agent for restaurant bookings using Ollama."""
    
    def __init__(
        self, 
        api_client: BookingAPIClient, 
        model_name: str = "llama3.2:3b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434"
    ):
        self.api_client = api_client
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            timeout=30
        )
        
        # Store session data for context management
        self.session_data: Dict[str, Any] = {}
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """Process a user message and return the agent's response."""
        try:
            # Initialize session if needed
            if session_id and session_id not in self.session_data:
                self.session_data[session_id] = {
                    'booking_info': {},
                    'conversation_history': []
                }
            
            session = self.session_data[session_id] if session_id else {}
            
            # Extract information from the message
            extracted_info = self._extract_booking_info(message)
            
            # Update session with extracted info
            if session_id:
                session['booking_info'].update(extracted_info)
                session['conversation_history'].append({
                    'user': message,
                    'extracted': extracted_info
                })
            
            # Determine the appropriate response
            response = self._generate_response(message, extracted_info, session)
            
            # Store response in session
            if session_id:
                session['conversation_history'][-1]['agent'] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._fallback_response(message)
    
    def _extract_booking_info(self, message: str) -> Dict[str, Any]:
        """Extract booking information from the message."""
        info = {}
        message_lower = message.lower()
        
        # Extract name
        name_patterns = [
            r"(?:my name is|i am|i'm|it's|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$",
            r"(?:for|under the name of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in name_patterns:
            import re
            if match := re.search(pattern, message.strip(), re.I):
                name = match.group(1).strip()
                # Don't set name if it's clearly a date or time
                if not any(word in name.lower() for word in ['today', 'tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                    info['name'] = name
                    break
        
        # Extract date
        today = datetime.now()
        if "today" in message_lower:
            info['date'] = today.strftime('%Y-%m-%d')
        elif "tomorrow" in message_lower:
            info['date'] = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif "weekend" in message_lower:
            days_ahead = 5 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            info['date'] = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
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
                    info['date'] = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                    break
        
        # Extract time
        time_patterns = [
            r'(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>[ap]m)?',
            r'(?P<hour>\d{1,2})\s*(?P<ampm>[ap]m)',
            r'\b(?P<hour>\d{1,2})\b(?!\s*(?:people?|guests?|persons?))'
        ]
        
        for pattern in time_patterns:
            import re
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
            import re
            if match := re.search(pattern, message_lower):
                num = int(match.group(1))
                if 1 <= num <= 20:
                    info['party_size'] = num
                    break
        
        return info
    
    def _generate_response(self, message: str, extracted_info: Dict[str, Any], session: Dict[str, Any]) -> str:
        """Generate appropriate response based on extracted information."""
        
        # Check if we have enough info for a booking
        booking_info = session.get('booking_info', {})
        all_info = {**booking_info, **extracted_info}
        
        # If we have all required info, try to create booking
        if all_info.get('name') and all_info.get('date') and all_info.get('time') and all_info.get('party_size'):
            return self._create_booking(all_info)
        
        # If we have date and time, check availability
        if all_info.get('date') and all_info.get('time'):
            return self._check_availability(all_info)
        
        # If we have date, check availability
        if all_info.get('date'):
            return self._check_availability(all_info)
        
        # Otherwise, ask for missing information
        return self._ask_for_missing_info(all_info)
    
    def _create_booking(self, info: Dict[str, Any]) -> str:
        """Create a booking with the provided information."""
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
    
    def _check_availability(self, info: Dict[str, Any]) -> str:
        """Check availability for the given date."""
        try:
            date = info.get('date', datetime.now().strftime('%Y-%m-%d'))
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
                    return f"❌ No availability for {date}. Would you like to try another date?"
            return "❌ Couldn't check availability. Please try again."
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return "❌ Error checking availability. Please try again."
    
    def _ask_for_missing_info(self, info: Dict[str, Any]) -> str:
        """Ask for missing booking information."""
        missing = []
        if not info.get('name'):
            missing.append("your name")
        if not info.get('date'):
            missing.append("the date")
        if not info.get('time'):
            missing.append("the time")
        if not info.get('party_size'):
            missing.append("number of people")
        
        if len(missing) == 4:
            return """🎯 I'd be happy to help you make a reservation! 

To get started, I'll need a few details:
• Your name
• Date you'd like to visit (e.g., "tomorrow", "saturday", "2025-08-15")
• Time you prefer (e.g., "7pm", "19:30")
• Number of people in your party

What's your name?"""
        elif 'name' in missing:
            return "What's your name for the reservation?"
        elif 'date' in missing:
            return "What date would you like to visit? (e.g., 'tomorrow', 'saturday', 'next friday')"
        elif 'time' in missing:
            return "What time would you prefer? (e.g., '7pm', '19:30')"
        elif 'party_size' in missing:
            return "How many people will be in your party?"
        else:
            return f"To make a booking, I need {', '.join(missing)}."
    
    def _fallback_response(self, message: str) -> str:
        """Fallback response when processing fails."""
        message_lower = message.lower()
        
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
        
        else:
            return """🍽️ Welcome to TheHungryUnicorn! I can help you with:

• 📅 Check availability for any date
• 🎯 Make a new reservation
• 📋 View your existing booking details
• ✏️ Modify or cancel your reservation

What would you like to do today?"""
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session."""
        if session_id and session_id in self.session_data:
            del self.session_data[session_id]
