"""LLM-powered booking agent that actually uses Ollama for understanding."""

import json
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
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
            timeout=30
        )
        self.sessions = {}
        
        # Create prompts for different tasks
        self.intent_prompt = PromptTemplate(
            input_variables=["message", "context"],
            template="""You are analyzing a message from a restaurant booking system user.
            
Current context: {context}
User message: {message}

Identify the user's intent. Choose EXACTLY ONE from:
- greeting (user is saying hello)
- check_availability (wants to see available times)
- create_booking (wants to make a reservation)
- check_booking (wants to see existing booking details)
- update_booking (wants to modify a booking)
- cancel_booking (wants to cancel)
- provide_info (user is providing information like name, date, time)
- unclear (intent is not clear)

Respond with ONLY the intent, nothing else."""
        )
        
        self.extraction_prompt = PromptTemplate(
            input_variables=["message", "current_context"],
            template="""Extract booking information from this message. Today is {today}.

Current context: {current_context}
Message: {message}

Extract the following if present:
1. Name (person's full name)
2. Date (convert to YYYY-MM-DD format)
3. Time (convert to HH:MM:SS format, assume PM for dinner times 5-11)
4. Party size (number of people)
5. Booking reference (like ABC1234)

Important date conversions:
- "tomorrow" = {tomorrow}
- "this weekend" or "saturday" = {saturday}
- "sunday" = {sunday}

Respond in JSON format ONLY:
{{"name": null or "value", "date": null or "YYYY-MM-DD", "time": null or "HH:MM:SS", "party_size": null or number, "booking_id": null or "ID"}}"""
        )
        
        self.response_prompt = PromptTemplate(
            input_variables=["context", "intent", "api_result"],
            template="""You are a friendly restaurant booking assistant for TheHungryUnicorn.

Context: {context}
User intent: {intent}
API result (if any): {api_result}

Generate a natural, friendly response. Use emojis where appropriate (no horse emojis).
Be concise but helpful. If information is missing for a booking, ask for it naturally.
When asking for a name make sure to sanity check the name.
If showing availability, list the times clearly.
If confirming a booking, show all the details clearly, including Name, Date, Time, Party Size and Booking Reference. 
MAKE SURE THESE ARE DEFINITELY OUTLINED IN THE CONFIRMATION.
Not when asking for but ONLY when responding with use:
-their name say their full name.
-the date of their booking do it in day then date form (Thursday 13th August 2025).
-the time give it as 12 hour clock (5:00pm or 5:30pm).



Response:"""
        )

    def clear_memory(self, session_id: str):
        """Clear session memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _get_date_context(self) -> Dict:
        """Get current date context for the LLM."""
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        # Calculate this Saturday
        days_to_saturday = (5 - today.weekday()) % 7
        if days_to_saturday == 0:
            days_to_saturday = 7
        saturday = today + timedelta(days=days_to_saturday)
        
        # Calculate this Sunday
        days_to_sunday = (6 - today.weekday()) % 7
        if days_to_sunday == 0:
            days_to_sunday = 7
        sunday = today + timedelta(days=days_to_sunday)
        
        return {
            'today': today.strftime('%Y-%m-%d'),
            'tomorrow': tomorrow.strftime('%Y-%m-%d'),
            'saturday': saturday.strftime('%Y-%m-%d'),
            'sunday': sunday.strftime('%Y-%m-%d')
        }

    def _extract_info_with_llm(self, message: str, context: Dict) -> Dict:
        """Use LLM to extract information from the message."""
        try:
            date_context = self._get_date_context()
            
            # Create the extraction prompt with current context
            prompt = self.extraction_prompt.format(
                message=message,
                current_context=json.dumps(context),
                **date_context
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            
            # Parse the JSON response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Clean up the response to get just the JSON
            content = content.strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            # Parse JSON
            extracted = json.loads(content)
            
            # Update context with non-null values
            for key, value in extracted.items():
                if value is not None and value != "null":
                    context[key] = value
            
            logger.info(f"LLM extracted: {extracted}")
            return context
            
        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            # Fall back to simple extraction if LLM fails
            return self._simple_extract(message, context)

    def _simple_extract(self, message: str, context: Dict) -> Dict:
        """Simple fallback extraction."""
        msg = message.lower()
        
        # Check if message looks like just a name
        if len(message.split()) <= 3 and not any(word in msg for word in ['book', 'table', 'cancel', 'check']):
            context['name'] = message.title()
        
        # Check for numbers that might be party size
        import re
        if match := re.search(r'\b(\d+)\s*(?:people|persons?|guests?)\b', msg):
            context['party_size'] = int(match.group(1))
        
        return context

    def _detect_intent_with_llm(self, message: str, context: Dict) -> str:
        """Use LLM to detect user intent."""
        try:
            prompt = self.intent_prompt.format(
                message=message,
                context=json.dumps(context)
            )
            
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                intent = response.content.strip().lower()
            else:
                intent = str(response).strip().lower()
            
            logger.info(f"LLM detected intent: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Error in intent detection: {e}")
            # Fallback to simple detection
            msg = message.lower()
            if 'book' in msg or 'reservation' in msg:
                return 'create_booking'
            elif 'availability' in msg or 'available' in msg:
                return 'check_availability'
            return 'unclear'

    def _generate_response_with_llm(self, context: Dict, intent: str, api_result: str = None) -> str:
        """Use LLM to generate a natural response."""
        try:
            prompt = self.response_prompt.format(
                context=json.dumps(context),
                intent=intent,
                api_result=api_result or "None"
            )
            
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I understand you'd like help with a booking. Could you please tell me more?"

    def process_message(self, message: str, session_id: str) -> str:
        """Process user message using LLM for understanding."""
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'context': {},
                'conversation_history': []
            }
        
        session = self.sessions[session_id]
        context = session['context']
        
        # Add to conversation history
        session['conversation_history'].append({'role': 'user', 'content': message})
        
        # Use LLM to extract information
        context = self._extract_info_with_llm(message, context)
        
        # Use LLM to detect intent
        intent = self._detect_intent_with_llm(message, context)
        
        # Handle different intents
        api_result = None
        
        if intent == 'greeting':
            response = self._generate_response_with_llm(context, intent)
            
        elif intent == 'check_availability':
            if context.get('date'):
                api_result = self._check_availability(context)
            response = self._generate_response_with_llm(context, intent, api_result)
            
        elif intent in ['create_booking', 'provide_info']:
            # Check if we have all required info
            required = ['name', 'date', 'time', 'party_size']
            missing = [field for field in required if not context.get(field)]
            
            if not missing:
                # We have everything, create the booking
                api_result = self._create_booking(context)
                response = self._generate_response_with_llm(context, 'booking_confirmed', api_result)
                # Clear context after successful booking
                if 'success' in str(api_result):
                    context.clear()
            else:
                # Ask for missing information
                response = self._generate_response_with_llm(context, 'need_more_info', f"Missing: {missing}")
                
        elif intent == 'check_booking':
            if context.get('booking_id'):
                api_result = self._get_booking(context)
            response = self._generate_response_with_llm(context, intent, api_result)
            
        elif intent == 'cancel_booking':
            if context.get('booking_id'):
                api_result = self._cancel_booking(context)
            response = self._generate_response_with_llm(context, intent, api_result)
            
        elif intent == 'update_booking':
            if context.get('booking_id'):
                api_result = self._update_booking(context)
            response = self._generate_response_with_llm(context, intent, api_result)
            
        else:
            response = self._generate_response_with_llm(context, intent)
        
        # Add response to history
        session['conversation_history'].append({'role': 'assistant', 'content': response})
        
        # Keep history manageable
        if len(session['conversation_history']) > 20:
            session['conversation_history'] = session['conversation_history'][-20:]
        
        return response

    def _check_availability(self, context: Dict) -> str:
        """Check availability and return result."""
        date = context.get('date')
        result = self.api_client.check_availability(date)
        
        if result['success']:
            slots = result.get('data', {}).get('available_slots', [])
            if slots:
                times = [s['time'] if isinstance(s, dict) else str(s) for s in slots[:8]]
                return f"Available times for {date}: {', '.join(times)}"
            return f"No availability for {date}"
        return f"Error checking availability: {result.get('error', 'Unknown')}"

    def _create_booking(self, context: Dict) -> str:
        """Create a booking and return result."""
        result = self.api_client.create_booking(
            customer_name=context['name'],
            date=context['date'],
            time=context['time'],
            party_size=context['party_size']
        )
        
        if result['success']:
            booking_data = result['data']
            return f"Booking confirmed! Reference: {booking_data.get('booking_reference')}. Details: {json.dumps(booking_data)}"
        return f"Booking failed: {result.get('error', 'Unknown error')}"

    def _get_booking(self, context: Dict) -> str:
        """Get booking details."""
        result = self.api_client.get_booking(context['booking_id'])
        if result['success']:
            return f"Booking found: {json.dumps(result['data'])}"
        return f"Booking not found: {context['booking_id']}"

    def _cancel_booking(self, context: Dict) -> str:
        """Cancel a booking."""
        result = self.api_client.cancel_booking(context['booking_id'])
        if result['success']:
            return f"Booking {context['booking_id']} cancelled successfully"
        return f"Failed to cancel booking: {result.get('error', 'Unknown')}"

    def _update_booking(self, context: Dict) -> str:
        """Update a booking."""
        updates = {}
        if context.get('date'):
            updates['date'] = context['date']
        if context.get('time'):
            updates['time'] = context['time']
        if context.get('party_size'):
            updates['party_size'] = context['party_size']
        
        if updates:
            result = self.api_client.update_booking(context['booking_id'], **updates)
            if result['success']:
                return f"Booking {context['booking_id']} updated successfully"
            return f"Failed to update: {result.get('error', 'Unknown')}"
        return "No updates specified"