"""Concise booking agent optimized for Ollama."""

import re
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from langchain_community.chat_models import ChatOllama
from booking_client import BookingAPIClient

logger = logging.getLogger(__name__)


class BookingAgent:
    def __init__(self, api_client: BookingAPIClient, model: str = "llama3.2:3b"):
        self.api_client = api_client
        self.llm = ChatOllama(model=model, temperature=0.1, timeout=30)
        self.sessions = {}

    def clear_memory(self, session_id: str):
        """Clear session memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _extract_info(self, message: str, context: Dict):
        """Extract time, date, name, party size, and booking ID from the message."""
        msg = message.lower()

        # Party size
        if match := re.search(r'\b(\d+)\b', message):
            num = int(match.group(1))
            if 1 <= num <= 20:
                context['party_size'] = num

        # Time (safe matching with error handling)
        time_patterns = [
            r'(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>[ap]m)?',  # 7:30pm
            r'(?P<hour>\d{1,2})\s*(?P<ampm>[ap]m)',                     # 7pm
            r'\b(?P<hour>\d{1,2})\b'                                    # 7
        ]

        for pattern in time_patterns:
            if match := re.search(pattern, msg):
                try:
                    hour = int(match.group("hour"))
                    minute = int(match.group("minute")) if "minute" in match.groupdict() and match.group("minute") else 0
                    meridiem = match.group("ampm") if "ampm" in match.groupdict() else None

                    if meridiem == 'pm' and hour < 12:
                        hour += 12
                    elif meridiem == 'am' and hour == 12:
                        hour = 0

                    if 10 <= hour <= 23:
                        context['time'] = f"{hour:02d}:{minute:02d}:00"
                        break
                except ValueError:
                    continue  # Skip invalid matches

        # Date (today, tomorrow, weekday)
        today = datetime.now()
        if "today" in msg:
            context['date'] = today.strftime('%Y-%m-%d')
        elif "tomorrow" in msg:
            context['date'] = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            weekdays = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
            }
            for day, idx in weekdays.items():
                if day in msg:
                    days_ahead = (idx - today.weekday()) % 7
                    context['date'] = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                    break

        # Name
        name_patterns = [
            r"(?:my name is|i am|i'm|it's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$"
        ]
        for pattern in name_patterns:
            if match := re.search(pattern, message.strip(), re.I):
                context['name'] = match.group(1).strip()
                break

        # Booking ID
        if match := re.search(r'\b([A-Z0-9]{6,8})\b', message):
            context['booking_id'] = match.group(1)

    
    def process_message(self, message: str, session_id: str) -> str:
        """Process user message and return response."""
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = {'booking_flow': False}
        
        context = self.sessions[session_id]
        
        # Extract information from message
        self._extract_info(message, context)
        
        # Check for clear intent changes first
        lower_msg = message.lower()
        
        if 'cancel' in lower_msg and 'reservation' in lower_msg:
            context['booking_flow'] = False
            return self._cancel_booking(context)
        elif any(phrase in lower_msg for phrase in ['check my booking', 'check my reservation', 'my booking', 'my reservation']):
            context['booking_flow'] = False
            return self._get_booking(context)
        elif any(word in lower_msg for word in ['availability', 'available']):
            context['booking_flow'] = False
            return self._check_availability(context)
        elif any(phrase in lower_msg for phrase in ['book a table', 'make a reservation', 'book table', 'reservation for']):
            context['booking_flow'] = True
            return self._create_booking(context)
        elif any(word in lower_msg for word in ['change', 'modify', 'update']) and 'booking' in lower_msg:
            context['booking_flow'] = False
            return self._update_booking(context)
        else:
            # If we're in a booking flow, continue it
            if context.get('booking_flow'):
                return self._create_booking(context)
            
            # Check if this might be a response to a previous question
            if message.strip().isdigit() and context.get('date'):
                # User selected a time from availability list
                context['booking_flow'] = True
                return self._create_booking(context)
            
            # Default greeting
            return "Hello! I can help you book a table, check availability, or manage existing reservations. What would you like to do?"
    
    def _check_availability(self, context: Dict) -> str:
        """Check availability."""
        date = context.get('date', datetime.now().strftime('%Y-%m-%d'))
        result = self.api_client.check_availability(date)
        
        if result['success']:
            slots = result.get('data', {}).get('available_slots', [])
            if slots:
                times = [s['time'].replace(':00', '') if isinstance(s, dict) else str(s).replace(':00', '') 
                        for s in slots[:6]]
                return f"Available times for {date}:\n" + "\n".join(f"• {t}" for t in times)
            return f"No availability for {date}. Try another date?"
        return "Couldn't check availability. Please try again."
    
    def _create_booking(self, context: Dict) -> str:
        """Create a booking."""
        missing = []
        if 'name' not in context:
            missing.append("your name")
        if 'date' not in context:
            missing.append("the date")
        if 'time' not in context:
            missing.append("the time")
        if 'party_size' not in context:
            missing.append("number of people")
        
        if missing:
            return f"To make a booking, I need {', '.join(missing)}."
        
        result = self.api_client.create_booking(
            customer_name=context['name'],
            date=context['date'],
            time=context['time'],
            party_size=context['party_size']
        )
        
        if result['success']:
            booking_id = result['data'].get('booking_id')
            time_display = context['time'].replace(':00', '')
            return f"""✅ Booking confirmed!
Reference: {booking_id}
Name: {context['name']}
Date: {context['date']}
Time: {time_display}
Party: {context['party_size']} people"""
        return "Couldn't create booking. Please try again."
    
    def _get_booking(self, context: Dict) -> str:
        """Get booking details."""
        if 'booking_id' not in context:
            return "Please provide your booking reference."
        
        result = self.api_client.get_booking(context['booking_id'])
        if result['success']:
            data = result['data']
            return f"""Booking {context['booking_id']}:
Name: {data.get('customer_name')}
Date: {data.get('date')}
Time: {data.get('time', '').replace(':00', '')}
Party: {data.get('party_size')} people"""
        return f"Couldn't find booking {context['booking_id']}"
    
    def _cancel_booking(self, context: Dict) -> str:
        """Cancel a booking."""
        if 'booking_id' not in context:
            return "Please provide your booking reference to cancel."
        
        result = self.api_client.cancel_booking(context['booking_id'])
        if result['success']:
            return f"✅ Booking {context['booking_id']} cancelled."
        return "Couldn't cancel booking."
    
    def _update_booking(self, context: Dict) -> str:
        """Update a booking."""
        if 'booking_id' not in context:
            return "Please provide your booking reference to update."
        
        updates = {}
        if 'date' in context:
            updates['date'] = context['date']
        if 'time' in context:
            updates['time'] = context['time']
        if 'party_size' in context:
            updates['party_size'] = context['party_size']
        
        if not updates:
            return "What would you like to change? (date, time, or party size)"
        
        result = self.api_client.update_booking(context['booking_id'], **updates)
        if result['success']:
            return f"✅ Booking {context['booking_id']} updated."
        return "Couldn't update booking."
