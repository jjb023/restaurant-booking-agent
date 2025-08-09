"""Simplified LLM-powered booking agent using Ollama."""

import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from langchain_community.chat_models import ChatOllama
from booking_client import BookingAPIClient

logger = logging.getLogger(__name__)


class BookingAgent:
    def __init__(self, api_client: BookingAPIClient, model: str = "llama3.2:3b", 
                 temperature: float = 0.1, base_url: str = "http://localhost:11434"):
        self.api_client = api_client
        self.llm = ChatOllama(
            model=model, 
            temperature=temperature, 
            base_url=base_url,
            timeout=60
        )
        self.sessions = {}

    def clear_memory(self, session_id: str):
        """Clear session memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _get_date_strings(self) -> Dict[str, str]:
        """Get helpful date strings for the LLM."""
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        # Calculate this weekend
        days_to_saturday = (5 - today.weekday()) % 7
        if days_to_saturday == 0:
            days_to_saturday = 7
        saturday = today + timedelta(days=days_to_saturday)
        
        days_to_sunday = (6 - today.weekday()) % 7
        if days_to_sunday == 0:
            days_to_sunday = 7
        sunday = today + timedelta(days=days_to_sunday)
        
        # Format dates nicely
        return {
            'today': today.strftime('%Y-%m-%d'),
            'tomorrow': tomorrow.strftime('%Y-%m-%d'),
            'saturday': saturday.strftime('%Y-%m-%d'),
            'sunday': sunday.strftime('%Y-%m-%d'),
            'today_formatted': today.strftime('%A, %B %d, %Y'),
            'tomorrow_formatted': tomorrow.strftime('%A, %B %d, %Y'),
            'saturday_formatted': saturday.strftime('%A, %B %d, %Y'),
            'sunday_formatted': sunday.strftime('%A, %B %d, %Y')
        }

    def _understand_message(self, message: str, context: Dict) -> Dict[str, Any]:
        """Use LLM to understand the user's message and extract information."""
        dates = self._get_date_strings()
        
        prompt = f"""You are helping understand a restaurant booking request. Extract information from the user's message.

Today is {dates['today_formatted']} ({dates['today']})
Tomorrow is {dates['tomorrow_formatted']} ({dates['tomorrow']})
This Saturday is {dates['saturday_formatted']} ({dates['saturday']})
This Sunday is {dates['sunday_formatted']} ({dates['sunday']})

Current booking context:
{json.dumps(context, indent=2)}

User message: "{message}"

Extract the following information if present:
1. intent: What does the user want? (check_availability, make_booking, check_booking, update_booking, cancel_booking, greeting, provide_info)
2. name: Full name if provided
3. date: Date in YYYY-MM-DD format
4. time: Time in HH:MM format (assume PM for 5-11, use 24-hour format)
5. party_size: Number of people
6. booking_reference: Booking ID if mentioned (format: 6-7 alphanumeric characters like ABC1234)
7. special_requests: Any special requests

Important:
- For times like "7pm" convert to "19:00"
- For times like "7:30pm" convert to "19:30"
- For dates like "tomorrow" use {dates['tomorrow']}
- For dates like "this weekend" or "saturday" use {dates['saturday']}
- For dates like "next Friday" calculate the correct date
- If the user provides just a name (like "John Smith"), set intent as "provide_info"
- If they say a number of people (like "4" or "4 people"), extract party_size

Respond ONLY with a JSON object, nothing else:
{{"intent": "...", "name": "...", "date": "...", "time": "...", "party_size": ..., "booking_reference": "...", "special_requests": "..."}}

Use null for any field not found in the message."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and parse JSON
            content = content.strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            extracted = json.loads(content)
            logger.info(f"Extracted: {extracted}")
            return extracted
            
        except Exception as e:
            logger.error(f"Error understanding message: {e}")
            return {"intent": "unclear"}

    def _generate_response(self, intent: str, context: Dict, api_result: Optional[Dict] = None) -> str:
        """Generate a natural response based on intent and context."""
        dates = self._get_date_strings()
        
        # Build the prompt based on the situation
        if intent == "greeting":
            return "Hello! 👋 Welcome to TheHungryUnicorn! I can help you make a reservation, check availability, or manage existing bookings. What would you like to do today?"
        
        elif intent == "check_availability" and api_result and api_result.get('success'):
            slots = api_result.get('data', {}).get('available_slots', [])
            if slots:
                date = context.get('date', 'the selected date')
                times = []
                for slot in slots[:10]:  # Show max 10 slots
                    time_str = slot.get('time', slot) if isinstance(slot, dict) else str(slot)
                    # Convert to 12-hour format for display
                    try:
                        time_obj = datetime.strptime(time_str.split(':')[0] + ':' + time_str.split(':')[1], '%H:%M')
                        times.append(time_obj.strftime('%-I:%M %p').lower())
                    except:
                        times.append(time_str)
                
                return f"Great! I found available times for {date}:\n\n" + \
                       "• " + "\n• ".join(times) + \
                       "\n\nWhich time would work best for you?"
            else:
                return f"I'm sorry, but we don't have any tables available on {context.get('date')}. Would you like to check another date?"
        
        elif intent == "booking_confirmed" and api_result and api_result.get('success'):
            booking_data = api_result.get('data', {})
            booking_ref = booking_data.get('booking_reference', 'N/A')
            
            # Format the date nicely
            date_str = context.get('date', '')
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%A, %B %d, %Y')
            except:
                formatted_date = date_str
            
            # Format the time nicely
            time_str = context.get('time', '')
            try:
                time_obj = datetime.strptime(time_str, '%H:%M:%S')
                formatted_time = time_obj.strftime('%-I:%M %p').lower()
            except:
                formatted_time = time_str
            
            return f"""🎉 Perfect! Your reservation is confirmed!

**Booking Details:**
📋 **Booking Reference:** {booking_ref}
👤 **Name:** {context.get('name', 'Guest')}
📅 **Date:** {formatted_date}
🕐 **Time:** {formatted_time}
👥 **Party Size:** {context.get('party_size', 2)} people

Please save your booking reference ({booking_ref}) - you'll need it to check or modify your reservation.

See you soon at TheHungryUnicorn! 🦄"""
        
        elif intent in ["make_booking", "provide_info"]:
            # Check what's missing
            required = {
                'name': 'your name',
                'date': 'the date you\'d like to visit',
                'time': 'your preferred time',
                'party_size': 'the number of people'
            }
            
            missing = []
            for field, description in required.items():
                if not context.get(field):
                    missing.append(description)
            
            if missing:
                if len(missing) == 4:  # Nothing provided yet
                    return "I'd be happy to help you make a reservation! To get started, could you tell me:\n• Your name\n• When you'd like to visit (date)\n• What time you prefer\n• How many people will be dining?"
                elif len(missing) == 1:
                    return f"Great! I just need {missing[0]} to complete your booking."
                else:
                    return f"Thanks! I still need:\n• " + "\n• ".join(missing)
            else:
                return "I have all your details! Let me create that booking for you..."
        
        elif intent == "check_booking":
            if not context.get('booking_reference'):
                return "I'd be happy to check your booking! Could you please provide your booking reference? It should be a 6-7 character code like ABC1234."
            elif api_result and api_result.get('success'):
                data = api_result.get('data', {})
                return f"""Found your booking!

📋 **Booking Reference:** {data.get('booking_reference')}
👤 **Name:** {data.get('customer', {}).get('first_name', '')} {data.get('customer', {}).get('surname', '')}
📅 **Date:** {data.get('visit_date')}
🕐 **Time:** {data.get('visit_time')}
👥 **Party Size:** {data.get('party_size')} people

Would you like to modify or cancel this booking?"""
        
        elif intent == "cancel_booking":
            if not context.get('booking_reference'):
                return "To cancel a booking, I'll need your booking reference. It should be a 6-7 character code like ABC1234."
            elif api_result and api_result.get('success'):
                return f"✅ Your booking (reference: {context.get('booking_reference')}) has been successfully cancelled. Is there anything else I can help you with?"
        
        elif intent == "update_booking":
            if not context.get('booking_reference'):
                return "To update a booking, I'll need your booking reference first. It should be a 6-7 character code like ABC1234."
            else:
                return "What would you like to change about your booking? You can update the date, time, or number of people."
        
        # Default response
        return "I'm here to help with restaurant bookings! You can:\n• Check availability for a date\n• Make a new reservation\n• Check an existing booking (with reference)\n• Modify or cancel a booking\n\nWhat would you like to do?"

    def process_message(self, message: str, session_id: str) -> str:
        """Process a user message and return a response."""
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'context': {},
                'history': []
            }
        
        session = self.sessions[session_id]
        context = session['context']
        
        # Add to history
        session['history'].append({'role': 'user', 'content': message})
        
        # Understand the message
        understanding = self._understand_message(message, context)
        intent = understanding.get('intent', 'unclear')
        
        # Update context with new information (skip null values)
        for key, value in understanding.items():
            if value is not None and key != 'intent':
                context[key] = value
        
        logger.info(f"Intent: {intent}, Context: {context}")
        
        # Handle different intents
        api_result = None
        
        if intent == "check_availability":
            if context.get('date'):
                # Call API to check availability
                api_result = self.api_client.check_availability(
                    date=context['date'],
                    party_size=context.get('party_size', 2)
                )
            response = self._generate_response(intent, context, api_result)
        
        elif intent in ["make_booking", "provide_info"]:
            # Check if we have all required info
            required = ['name', 'date', 'time', 'party_size']
            if all(context.get(field) for field in required):
                # Make the booking
                api_result = self.api_client.create_booking(
                    customer_name=context['name'],
                    date=context['date'],
                    time=context['time'],
                    party_size=context['party_size'],
                    special_requests=context.get('special_requests')
                )
                
                # Store booking reference if successful
                if api_result.get('success'):
                    booking_ref = api_result.get('data', {}).get('booking_reference')
                    if booking_ref:
                        context['last_booking_reference'] = booking_ref
                    response = self._generate_response("booking_confirmed", context, api_result)
                    # Clear context for next booking
                    session['context'] = {'last_booking_reference': booking_ref}
                else:
                    response = f"I'm sorry, there was an issue creating your booking: {api_result.get('error', 'Unknown error')}. Please try again."
            else:
                response = self._generate_response(intent, context)
        
        elif intent == "check_booking":
            if context.get('booking_reference'):
                api_result = self.api_client.get_booking(context['booking_reference'])
            response = self._generate_response(intent, context, api_result)
        
        elif intent == "cancel_booking":
            # If no reference provided, check if we have the last one
            if not context.get('booking_reference') and context.get('last_booking_reference'):
                context['booking_reference'] = context['last_booking_reference']
            
            if context.get('booking_reference'):
                api_result = self.api_client.cancel_booking(context['booking_reference'])
            response = self._generate_response(intent, context, api_result)
        
        elif intent == "update_booking":
            # Handle updates - for now just acknowledge
            response = self._generate_response(intent, context, api_result)
        
        else:
            response = self._generate_response(intent, context)
        
        # Add response to history
        session['history'].append({'role': 'assistant', 'content': response})
        
        # Keep history manageable
        if len(session['history']) > 20:
            session['history'] = session['history'][-20:]
        
        return response