"""Simplified agent that works reliably with Ollama."""

from typing import List, Dict, Any, Optional
import logging
import re
import json
from datetime import datetime, timedelta
from booking_client import BookingAPIClient
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)


class SimpleBookingAgent:
    """Simplified booking agent that works well with Ollama."""
    
    def __init__(
        self, 
        api_client: BookingAPIClient, 
        llm_provider: str = "ollama",
        llm_model: str = None, 
        temperature: float = 0.1,
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.api_client = api_client
        self.llm_provider = llm_provider.lower()
        
        # Initialize LLM
        if self.llm_provider == "ollama":
            self.llm = ChatOllama(
                model=llm_model or "llama3.2:3b",
                temperature=temperature,
                base_url=ollama_base_url,
                format="json",  # Request JSON format
                timeout=30
            )
            logger.info(f"Using Ollama with model: {llm_model or 'llama3.2:3b'}")
        else:
            self.llm = ChatOpenAI(
                model=llm_model or "gpt-3.5-turbo",
                temperature=temperature
            )
            logger.info(f"Using OpenAI with model: {llm_model or 'gpt-3.5-turbo'}")
        
        # Conversation history
        self.conversations = {}
    
    def _parse_date(self, date_str: str) -> str:
        """Parse natural language dates."""
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
        
        date_str = date_str.lower().strip()
        today = datetime.now()
        
        if 'today' in date_str:
            return today.strftime('%Y-%m-%d')
        elif 'tomorrow' in date_str:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'weekend' in date_str:
            days_ahead = 5 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        # Try to find weekday names
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name, day_num in weekdays.items():
            if day_name in date_str:
                days_ahead = day_num - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                if 'next' in date_str:
                    days_ahead += 7
                return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        return date_str
    
    def _parse_time(self, time_str: str) -> str:
        """Parse natural language times."""
        if not time_str:
            return "19:00"
        
        time_str = time_str.lower().strip()
        
        # Extract time pattern
        match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*([ap]m)?', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            meridiem = match.group(3)
            
            if meridiem == 'pm' and hour < 12:
                hour += 12
            elif meridiem == 'am' and hour == 12:
                hour = 0
            
            return f"{hour:02d}:{minute:02d}"
        
        return time_str
    
    def _extract_intent_and_details(self, message: str) -> Dict[str, Any]:
        """Extract intent and details from the message using LLM."""
        prompt = f"""Analyze this restaurant booking request and extract the information as JSON.

Message: "{message}"

Return ONLY valid JSON with these fields (use null if not provided):
{{
  "intent": "check_availability" or "create_booking" or "get_booking" or "update_booking" or "cancel_booking" or "general",
  "customer_name": "name or null",
  "date": "date mentioned or null", 
  "time": "time mentioned or null",
  "party_size": number or null,
  "booking_id": "booking reference or null"
}}

Examples:
- "I want to book a table for 4 tomorrow at 7pm" -> {{"intent": "create_booking", "customer_name": null, "date": "tomorrow", "time": "7pm", "party_size": 4, "booking_id": null}}
- "Check availability for this weekend" -> {{"intent": "check_availability", "customer_name": null, "date": "this weekend", "time": null, "party_size": null, "booking_id": null}}
- "My name is John Smith" -> {{"intent": "general", "customer_name": "John Smith", "date": null, "time": null, "party_size": null, "booking_id": null}}
- "4" -> {{"intent": "general", "customer_name": null, "date": null, "time": null, "party_size": 4, "booking_id": null}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.error(f"No JSON found in response: {content}")
                return {"intent": "general"}
        except Exception as e:
            logger.error(f"Error extracting intent: {e}")
            
            # Fallback to simple pattern matching
            intent = "general"
            if any(word in message.lower() for word in ['book', 'reservation', 'table for']):
                intent = "create_booking"
            elif 'availability' in message.lower() or 'available' in message.lower():
                intent = "check_availability"
            elif 'cancel' in message.lower():
                intent = "cancel_booking"
            elif any(word in message.lower() for word in ['check', 'my booking', 'my reservation']):
                intent = "get_booking"
            
            # Extract details with regex
            details = {"intent": intent}
            
            # Extract party size
            party_match = re.search(r'\b(\d+)\s*(?:people|persons?|guests?)?\b', message)
            if party_match:
                details["party_size"] = int(party_match.group(1))
            
            # Extract time
            time_match = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b', message.lower())
            if time_match:
                details["time"] = time_match.group(0)
            
            # Extract booking ID
            booking_match = re.search(r'\b([A-Z0-9]{6,8})\b', message)
            if booking_match:
                details["booking_id"] = booking_match.group(1)
            
            return details
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """Process a user message and return a response."""
        try:
            # Initialize conversation history
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "history": [],
                    "context": {}
                }
            
            conversation = self.conversations[session_id]
            conversation["history"].append({"role": "user", "message": message})
            
            # Extract intent and details
            details = self._extract_intent_and_details(message)
            
            # Update context with new information
            for key in ["customer_name", "date", "time", "party_size", "booking_id"]:
                if details.get(key):
                    conversation["context"][key] = details[key]
            
            # Handle based on intent
            intent = details.get("intent", "general")
            
            if intent == "check_availability":
                return self._handle_check_availability(conversation["context"])
            
            elif intent == "create_booking":
                return self._handle_create_booking(conversation["context"], message)
            
            elif intent == "get_booking":
                return self._handle_get_booking(conversation["context"])
            
            elif intent == "cancel_booking":
                return self._handle_cancel_booking(conversation["context"])
            
            elif intent == "update_booking":
                return self._handle_update_booking(conversation["context"])
            
            else:
                # General conversation - check what info we need
                return self._handle_general_conversation(conversation["context"], message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, but I encountered an error. Could you please rephrase your request?"
    
    def _handle_check_availability(self, context: Dict) -> str:
        """Handle availability check."""
        date = self._parse_date(context.get("date", "today"))
        time = self._parse_time(context.get("time")) if context.get("time") else None
        party_size = context.get("party_size")
        
        result = self.api_client.check_availability(date, time, party_size)
        
        if result['success']:
            data = result.get('data', {})
            slots = data.get('available_slots', [])
            
            if slots:
                slot_list = "\n".join([f"  • {s['time'] if isinstance(s, dict) else s}" for s in slots[:8]])
                return f"✅ Available times for {date}:\n{slot_list}\n\nWould you like to book one of these times?"
            else:
                return f"❌ No available slots for {date}. Would you like to check another date?"
        else:
            return "I couldn't check availability. Please try again."
    
    def _handle_create_booking(self, context: Dict, original_message: str) -> str:
        """Handle booking creation."""
        # Check what's missing
        missing = []
        if not context.get("customer_name"):
            missing.append("your name")
        if not context.get("date"):
            missing.append("the date")
        if not context.get("time"):
            missing.append("the time")
        if not context.get("party_size"):
            missing.append("the number of people")
        
        if missing:
            return f"I'd be happy to make a reservation! I just need {', '.join(missing)}. Could you provide that?"
        
        # Parse date and time
        date = self._parse_date(context["date"])
        time = self._parse_time(context["time"])
        
        # Create booking
        result = self.api_client.create_booking(
            customer_name=context["customer_name"],
            date=date,
            time=time,
            party_size=context["party_size"],
            contact_number=context.get("contact_number"),
            special_requests=context.get("special_requests")
        )
        
        if result['success']:
            booking_id = result['data'].get('booking_id', 'N/A')
            return f"""🎉 Perfect! Your reservation is confirmed!

📋 **Booking Reference:** {booking_id}
👤 **Name:** {context["customer_name"]}
📅 **Date:** {date}
🕐 **Time:** {time}
👥 **Party size:** {context["party_size"]}

Please save your booking reference. See you soon at TheHungryUnicorn!"""
        else:
            return f"❌ I couldn't complete the booking: {result.get('error', 'Unknown error')}"
    
    def _handle_get_booking(self, context: Dict) -> str:
        """Handle booking lookup."""
        if not context.get("booking_id"):
            return "Please provide your booking reference number so I can look it up."
        
        result = self.api_client.get_booking(context["booking_id"])
        
        if result['success']:
            booking = result['data']
            return f"""📋 Your Booking Details:

**Reference:** {booking.get('booking_id')}
**Name:** {booking.get('customer_name')}
**Date:** {booking.get('date')}
**Time:** {booking.get('time')}
**Party Size:** {booking.get('party_size')}
**Status:** {booking.get('status', 'Confirmed')}"""
        else:
            return f"❌ Could not find booking: {context['booking_id']}"
    
    def _handle_cancel_booking(self, context: Dict) -> str:
        """Handle booking cancellation."""
        if not context.get("booking_id"):
            return "Please provide your booking reference number to cancel."
        
        result = self.api_client.cancel_booking(context["booking_id"])
        
        if result['success']:
            return f"✅ Booking {context['booking_id']} has been cancelled successfully."
        else:
            return f"❌ Could not cancel booking: {result.get('error')}"
    
    def _handle_update_booking(self, context: Dict) -> str:
        """Handle booking updates."""
        if not context.get("booking_id"):
            return "Please provide your booking reference number to update."
        
        # Implement update logic here
        return "To update your booking, please let me know what you'd like to change (date, time, or party size)."
    
    def _handle_general_conversation(self, context: Dict, message: str) -> str:
        """Handle general conversation."""
        # Check if it's just a number (party size)
        if message.strip().isdigit():
            context["party_size"] = int(message.strip())
            return self._handle_create_booking(context, message)
        
        # Check if it's a name
        if any(phrase in message.lower() for phrase in ["my name is", "i am", "i'm"]):
            # Name was likely extracted
            return "Nice to meet you! What can I help you with today?"
        
        # Default response
        return """Hello! I can help you with:
• Making a new reservation
• Checking availability
• Looking up your booking
• Cancelling a reservation

What would you like to do?"""
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory."""
        if session_id and session_id in self.conversations:
            del self.conversations[session_id]


# Create a wrapper for compatibility
class BookingAgent(SimpleBookingAgent):
    """Wrapper for compatibility with existing code."""
    pass