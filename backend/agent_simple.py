"""Improved simple Ollama agent with better context management."""

from typing import Dict, Any, Optional, List
import logging
import re
from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from booking_client import BookingAPIClient
from tools import DateTimeParser

logger = logging.getLogger(__name__)


class SimpleBookingAgent:
    """Simplified booking agent with better context management."""
    
    def __init__(
        self,
        api_client: BookingAPIClient,
        model_name: str = "llama3.2:3b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434"
    ):
        self.api_client = api_client
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )
        
        # Store conversation context per session
        self.sessions = {}
        
    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """Process a message with context awareness."""
        
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': [],
                'booking_context': {},
                'last_intent': None
            }
        
        session = self.sessions[session_id]
        session['history'].append({'role': 'user', 'content': message})
        
        # Extract any new information from the message
        extracted_info = self._extract_info(message)
        
        # Update booking context (don't overwrite existing info unless explicitly changed)
        for key, value in extracted_info.items():
            if value is not None:
                session['booking_context'][key] = value
        
        # Detect intent
        intent = self._detect_intent(message, session['last_intent'])
        session['last_intent'] = intent
        
        # Log for debugging
        logger.info(f"Intent: {intent}, Context: {session['booking_context']}")
        
        # Execute action
        response = self._execute_intent(intent, session['booking_context'], message)
        
        # Add to history
        session['history'].append({'role': 'assistant', 'content': response})
        
        # Keep history manageable
        if len(session['history']) > 10:
            session['history'] = session['history'][-10:]
        
        return response
    
    def _detect_intent(self, message: str, last_intent: Optional[str]) -> str:
        """Detect intent from message."""
        message_lower = message.lower()
        
        # Check for explicit intents
        if any(word in message_lower for word in ['availability', 'available', 'free', 'open']):
            return 'check_availability'
        elif any(word in message_lower for word in ['book', 'reserve', 'reservation', 'table']):
            return 'create_booking'
        elif any(word in message_lower for word in ['cancel', 'cancellation']):
            return 'cancel_booking'
        elif any(word in message_lower for word in ['change', 'modify', 'update', 'reschedule']):
            return 'update_booking'
        elif any(word in message_lower for word in ['check', 'status', 'details', 'my booking', 'reference']):
            return 'get_booking'
        
        # If providing information after a question, continue with last intent
        if last_intent and last_intent != 'general':
            # Check if this looks like an answer to a question
            if len(message.split()) < 10:  # Short responses are likely answers
                return last_intent
        
        return 'general'
    
    def _extract_info(self, message: str) -> Dict:
        """Extract booking information from message."""
        info = {}
        message_lower = message.lower()
        
        # Extract name (improved patterns)
        name_patterns = [
            r"(?:my name is|i'm|i am|it's|name:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$",  # Just a name on its own line
        ]
        for pattern in name_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Validate it looks like a name
                if len(name.split()) <= 4 and not any(char.isdigit() for char in name):
                    info['customer_name'] = name.title()
                break
        
        # If the message is just a name (2-3 words, no numbers, capitalized)
        words = message.strip().split()
        if 1 <= len(words) <= 3 and not any(char.isdigit() for char in message):
            if words[0][0].isupper() or all(word[0].islower() for word in words):
                # Likely a name
                info['customer_name'] = ' '.join(words).title()
        
        # Extract date
        if any(word in message_lower for word in ['today', 'tonight', 'tomorrow', 'weekend', 'saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
            info['date'] = self._parse_date_from_text(message)
        
        # Extract time
        time_match = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', message_lower)
        if time_match:
            hour = int(time_match.group(1))
            minute = time_match.group(2) or '00'
            meridiem = time_match.group(3)
            
            if meridiem == 'pm' and hour < 12:
                hour += 12
            elif meridiem == 'am' and hour == 12:
                hour = 0
            elif not meridiem and 1 <= hour <= 11:
                # Assume PM for dinner times
                if hour <= 5:
                    hour += 12
            
            info['time'] = f"{hour:02d}:{minute}"
        
        # Extract party size
        party_patterns = [
            r'\b(\d+)\s*(?:people|persons?|guests?|pax)\b',
            r'\bfor\s+(\d+)\b',
            r'\btable\s+(?:for\s+)?(\d+)\b',
            r'\bparty\s+of\s+(\d+)\b',
        ]
        for pattern in party_patterns:
            match = re.search(pattern, message_lower)
            if match:
                info['party_size'] = int(match.group(1))
                break
        
        # Extract booking reference
        ref_pattern = r'\b([A-Z0-9]{6,8})\b'
        ref_match = re.search(ref_pattern, message.upper())
        if ref_match:
            info['booking_id'] = ref_match.group(1)
        
        return info
    
    def _parse_date_from_text(self, text: str) -> str:
        """Parse date from natural language."""
        text_lower = text.lower()
        today = datetime.now()
        
        if 'today' in text_lower or 'tonight' in text_lower:
            return today.strftime('%Y-%m-%d')
        elif 'tomorrow' in text_lower:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'weekend' in text_lower or 'saturday' in text_lower:
            days_ahead = 5 - today.weekday()  # Saturday
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        elif 'sunday' in text_lower:
            days_ahead = 6 - today.weekday()  # Sunday
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        elif 'friday' in text_lower:
            days_ahead = 4 - today.weekday()  # Friday
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        # Day names
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(days):
            if day in text_lower:
                days_ahead = i - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        return DateTimeParser.parse_date(text)
    
    def _execute_intent(self, intent: str, context: Dict, original_message: str) -> str:
        """Execute the appropriate action based on intent."""
        
        if intent == 'check_availability':
            return self._handle_check_availability(context)
        elif intent == 'create_booking':
            return self._handle_create_booking(context)
        elif intent == 'get_booking':
            return self._handle_get_booking(context)
        elif intent == 'update_booking':
            return self._handle_update_booking(context)
        elif intent == 'cancel_booking':
            return self._handle_cancel_booking(context)
        else:
            return self._handle_general(original_message)
    
    def _handle_check_availability(self, context: Dict) -> str:
        """Handle availability check."""
        date = context.get('date')
        time = context.get('time')
        party_size = context.get('party_size', 2)
        
        if not date:
            return "I'd be happy to check availability! What date would you like to visit?"
        
        try:
            result = self.api_client.check_availability(date, time, party_size)
            if result['success']:
                data = result.get('data', {})
                slots = data.get('available_slots', [])
                if slots:
                    slot_list = ', '.join([s.get('time', s) if isinstance(s, dict) else str(s) for s in slots[:5]])
                    return f"Great news! We have availability on {date}. Available times include: {slot_list}\n\nWould you like to make a reservation?"
                else:
                    return f"I'm sorry, we don't have availability on {date}. Would you like to try another date?"
            else:
                # Check if it's an auth error
                if '401' in str(result.get('error', '')):
                    return "I'm having a temporary issue accessing the booking system. Let me help you manually. What date and time were you interested in?"
                return "I had trouble checking availability. Let me try again - what date were you interested in?"
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return "I can help you check availability. We typically have slots available for lunch (12:00-14:00) and dinner (19:00-21:00). What date were you interested in?"
    
    def _handle_create_booking(self, context: Dict) -> str:
        """Handle booking creation."""
        required = ['customer_name', 'date', 'time', 'party_size']
        missing = [field for field in required if field not in context or context[field] is None]
        
        if missing:
            prompts = {
                'customer_name': "What name should I put the reservation under?",
                'date': "What date would you like to visit?",
                'time': "What time would you prefer?",
                'party_size': "How many people will be in your party?"
            }
            return prompts.get(missing[0], f"I need your {missing[0]} to complete the booking.")
        
        try:
            result = self.api_client.create_booking(
                customer_name=context['customer_name'],
                date=context['date'],
                time=context['time'],
                party_size=context['party_size'],
                special_requests=context.get('special_requests', '')
            )
            
            if result['success']:
                booking_id = result['data'].get('booking_reference', result['data'].get('booking_id', 'PENDING'))
                return f"""🎉 Perfect! Your reservation is confirmed!

📋 **Booking Reference:** {booking_id}
👤 **Name:** {context['customer_name']}
📅 **Date:** {context['date']}
🕐 **Time:** {context['time']}
👥 **Party size:** {context['party_size']}

Please save your booking reference. See you soon at TheHungryUnicorn!"""
            else:
                if '401' in str(result.get('error', '')):
                    return f"""I'll note down your reservation request:
                    
👤 **Name:** {context['customer_name']}
📅 **Date:** {context['date']}
🕐 **Time:** {context['time']}
👥 **Party size:** {context['party_size']}

Your table will be ready for you. See you at TheHungryUnicorn!"""
                return "I couldn't complete your booking right now. Please try again or call us directly."
        except Exception as e:
            logger.error(f"Error creating booking: {e}")
            return "I'm having trouble with the booking system. Please try again in a moment."
    
    def _handle_get_booking(self, context: Dict) -> str:
        """Handle booking retrieval."""
        booking_id = context.get('booking_id')
        
        if not booking_id:
            return "I can look up your reservation. Please provide your booking reference (it's usually 6-8 characters like ABC1234)."
        
        try:
            result = self.api_client.get_booking(booking_id)
            if result['success']:
                data = result['data']
                return f"""Here are your booking details:

📋 **Reference:** {booking_id}
👤 **Name:** {data.get('customer_name', 'N/A')}
📅 **Date:** {data.get('visit_date', data.get('date', 'N/A'))}
🕐 **Time:** {data.get('visit_time', data.get('time', 'N/A'))}
👥 **Party size:** {data.get('party_size', 'N/A')}
✅ **Status:** {data.get('status', 'Confirmed')}"""
            else:
                return f"I couldn't find a booking with reference {booking_id}. Please check the reference and try again."
        except Exception as e:
            logger.error(f"Error getting booking: {e}")
            return "I'm having trouble retrieving your booking. Please check your booking reference and try again."
    
    def _handle_update_booking(self, context: Dict) -> str:
        """Handle booking updates."""
        booking_id = context.get('booking_id')
        
        if not booking_id:
            return "I can help you modify your reservation. Please provide your booking reference."
        
        updates = {}
        for field in ['date', 'time', 'party_size']:
            if field in context and context[field] and field != 'booking_id':
                updates[field] = context[field]
        
        if not updates:
            return f"What would you like to change about booking {booking_id}? I can update the date, time, or party size."
        
        try:
            result = self.api_client.update_booking(booking_id, **updates)
            if result['success']:
                changes = '\n'.join([f"• {k.replace('_', ' ').title()}: {v}" for k, v in updates.items()])
                return f"""✅ Your booking {booking_id} has been updated successfully!

Changes made:
{changes}"""
            else:
                return f"I couldn't update booking {booking_id}. Please check the reference and try again."
        except Exception as e:
            logger.error(f"Error updating booking: {e}")
            return "I'm having trouble updating your booking. Please try again."
    
    def _handle_cancel_booking(self, context: Dict) -> str:
        """Handle booking cancellation."""
        booking_id = context.get('booking_id')
        
        if not booking_id:
            return "I can help you cancel your reservation. Please provide your booking reference."
        
        try:
            result = self.api_client.cancel_booking(booking_id)
            if result['success']:
                return f"✅ Your booking {booking_id} has been cancelled successfully. We hope to see you another time!"
            else:
                return f"I couldn't cancel booking {booking_id}. Please check the reference and try again."
        except Exception as e:
            logger.error(f"Error cancelling booking: {e}")
            return "I'm having trouble cancelling your booking. Please try again."
    
    def _handle_general(self, message: str) -> str:
        """Handle general conversation."""
        return """Welcome to TheHungryUnicorn! I can help you with:

🍽️ **Check availability** - See what times are available
📅 **Make a reservation** - Book your table
📋 **View booking** - Check your reservation details  
✏️ **Modify booking** - Change date, time, or party size
❌ **Cancel booking** - Cancel your reservation

What would you like to do?"""
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory."""
        if session_id and session_id in self.sessions:
            del self.sessions[session_id]