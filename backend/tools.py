"""Improved LangChain tools for restaurant booking operations."""

from langchain.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import re
from booking_client import BookingAPIClient


class DateTimeParser:
    """Utility class for parsing natural language dates and times."""
    
    @staticmethod
    def parse_date(date_str: str) -> str:
        """Convert natural language date to YYYY-MM-DD format."""
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
            
        date_str = date_str.lower().strip()
        today = datetime.now()
        
        # Handle relative dates
        if 'today' in date_str:
            return today.strftime('%Y-%m-%d')
        elif 'tomorrow' in date_str:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'day after tomorrow' in date_str:
            return (today + timedelta(days=2)).strftime('%Y-%m-%d')
        elif 'weekend' in date_str or 'saturday' in date_str:
            # Next Saturday
            days_ahead = 5 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        elif 'sunday' in date_str:
            days_ahead = 6 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        # Handle weekdays
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
        
        # Try to parse standard date formats
        formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try to extract date components
        date_match = re.search(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?', date_str)
        if date_match:
            day_or_month = int(date_match.group(1))
            month_or_day = int(date_match.group(2))
            year = date_match.group(3)
            
            if year:
                year = int(year)
                if year < 100:
                    year += 2000
            else:
                year = today.year
            
            # Try both interpretations
            try:
                # Assume DD/MM/YYYY
                return datetime(year, month_or_day, day_or_month).strftime('%Y-%m-%d')
            except ValueError:
                try:
                    # Assume MM/DD/YYYY
                    return datetime(year, day_or_month, month_or_day).strftime('%Y-%m-%d')
                except ValueError:
                    pass
        
        return date_str  # Return as-is if parsing fails
    
    @staticmethod
    def parse_time(time_str: str) -> str:
        """Convert natural language time to HH:MM format."""
        if not time_str:
            return '19:00'  # Default dinner time
            
        time_str = time_str.lower().strip().replace('.', '').replace(' ', '')
        
        # Handle am/pm format
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*([ap]m)?', time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            meridiem = time_match.group(3)
            
            if meridiem == 'pm' and hour < 12:
                hour += 12
            elif meridiem == 'am' and hour == 12:
                hour = 0
            elif not meridiem and 1 <= hour <= 11:
                # Assume evening for restaurant times without AM/PM
                if hour <= 5:
                    hour += 12
            
            return f"{hour:02d}:{minute:02d}"
        
        return time_str


class CheckAvailabilityInput(BaseModel):
    date: str = Field(description="Date to check availability (e.g., 'next Friday', '2025-08-15', 'tomorrow')")
    time: Optional[str] = Field(None, description="Optional preferred time (e.g., '7pm', '19:00')")
    party_size: Optional[int] = Field(None, description="Number of people")


class CheckAvailabilityTool(BaseTool):
    name = "check_availability"
    description = "Check restaurant availability for a specific date and optionally time"
    args_schema: Type[BaseModel] = CheckAvailabilityInput
    api_client: BookingAPIClient = None
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, date: str, time: Optional[str] = None, party_size: Optional[int] = None) -> str:
        """Execute the availability check."""
        parsed_date = DateTimeParser.parse_date(date)
        parsed_time = DateTimeParser.parse_time(time) if time else None
        
        result = self.api_client.check_availability(parsed_date, parsed_time, party_size)
        
        if result['success']:
            data = result['data']
            if 'available_slots' in data:
                slots = data.get('available_slots', [])
                if slots:
                    slot_times = [slot['time'] if isinstance(slot, dict) else slot for slot in slots]
                    return f"✅ Available time slots for {parsed_date}:\n" + \
                           "\n".join([f"  • {time}" for time in slot_times[:8]])  # Limit to 8 slots
                else:
                    return f"❌ No available slots for {parsed_date}. Would you like to try another date?"
            return f"✅ Restaurant has availability on {parsed_date}" + (f" at {parsed_time}" if parsed_time else "")
        else:
            return f"❌ Error checking availability: {result.get('error', 'Unknown error')}"


class CreateBookingInput(BaseModel):
    customer_name: str = Field(description="Name of the customer making the reservation")
    date: str = Field(description="Date for the booking (e.g., 'tomorrow', '2025-08-15')")
    time: str = Field(description="Time for the booking (e.g., '7pm', '19:00')")
    party_size: int = Field(description="Number of people")
    contact_number: Optional[str] = Field(None, description="Contact phone number")
    special_requests: Optional[str] = Field(None, description="Any special requests or dietary requirements")


class CreateBookingTool(BaseTool):
    name = "create_booking"
    description = "Create a new restaurant booking reservation"
    args_schema: Type[BaseModel] = CreateBookingInput
    api_client: BookingAPIClient = None
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, customer_name: str, date: str, time: str, party_size: int,
             contact_number: Optional[str] = None, special_requests: Optional[str] = None) -> str:
        """Execute the booking creation."""
        # Validate customer name is not a tool name
        invalid_names = ['check', 'cancel', 'update', 'create', 'booking', 'reservation', 
                        'check availability', 'create booking', 'cancel booking']
        if customer_name.lower() in invalid_names:
            return "❌ Please provide a valid customer name for the reservation."
        
        parsed_date = DateTimeParser.parse_date(date)
        parsed_time = DateTimeParser.parse_time(time)
        
        result = self.api_client.create_booking(
            customer_name, parsed_date, parsed_time, party_size,
            contact_number, special_requests
        )
        
        if result['success']:
            booking = result['data']
            booking_ref = booking.get('booking_id', 'N/A')
            
            response = f"""🎉 Perfect! Your reservation is confirmed!

📋 **Booking Reference:** {booking_ref}
👤 **Name:** {customer_name}
📅 **Date:** {parsed_date}
🕐 **Time:** {parsed_time}
👥 **Party size:** {party_size}

Please save your booking reference. See you soon at TheHungryUnicorn!"""
            
            return response
        else:
            return f"❌ Failed to create booking: {result.get('error', 'Unknown error')}"


class GetBookingInput(BaseModel):
    booking_id: str = Field(description="The booking reference ID")


class GetBookingTool(BaseTool):
    name = "get_booking"
    description = "Retrieve details of an existing booking using the booking reference"
    args_schema: Type[BaseModel] = GetBookingInput
    api_client: BookingAPIClient = None
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, booking_id: str) -> str:
        """Execute the booking retrieval."""
        result = self.api_client.get_booking(booking_id)
        
        if result['success']:
            booking = result['data']
            response = f"""📋 Your Booking Details:

**Reference:** {booking.get('booking_id')}
**Name:** {booking.get('customer_name')}
**Date:** {booking.get('date')}
**Time:** {booking.get('time')}
**Party Size:** {booking.get('party_size')}
**Status:** {booking.get('status', 'Confirmed')}"""
            
            if booking.get('special_requests'):
                response += f"\n**Special Requests:** {booking.get('special_requests')}"
            
            return response
        else:
            return f"❌ Could not find booking with reference: {booking_id}"


class UpdateBookingInput(BaseModel):
    booking_id: str = Field(description="The booking reference ID")
    date: Optional[str] = Field(None, description="New date for the booking")
    time: Optional[str] = Field(None, description="New time for the booking")
    party_size: Optional[int] = Field(None, description="New party size")


class UpdateBookingTool(BaseTool):
    name = "update_booking"
    description = "Modify an existing booking (change date, time, or party size)"
    args_schema: Type[BaseModel] = UpdateBookingInput
    api_client: BookingAPIClient = None
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, booking_id: str, date: Optional[str] = None, 
             time: Optional[str] = None, party_size: Optional[int] = None) -> str:
        """Execute the booking update."""
        update_data = {}
        changes = []
        
        if date:
            parsed_date = DateTimeParser.parse_date(date)
            update_data['date'] = parsed_date
            changes.append(f"date to {parsed_date}")
        if time:
            parsed_time = DateTimeParser.parse_time(time)
            update_data['time'] = parsed_time
            changes.append(f"time to {parsed_time}")
        if party_size:
            update_data['party_size'] = party_size
            changes.append(f"party size to {party_size}")
        
        if not update_data:
            return "❌ No changes specified for the booking"
        
        result = self.api_client.update_booking(booking_id, **update_data)
        
        if result['success']:
            return f"""✅ Booking {booking_id} updated successfully!

Changed: {', '.join(changes)}

Your reservation has been modified as requested."""
        else:
            return f"❌ Failed to update booking: {result.get('error', 'Unknown error')}"


class CancelBookingInput(BaseModel):
    booking_id: str = Field(description="The booking reference ID to cancel")


class CancelBookingTool(BaseTool):
    name = "cancel_booking"
    description = "Cancel an existing restaurant booking"
    args_schema: Type[BaseModel] = CancelBookingInput
    api_client: BookingAPIClient = None
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, booking_id: str) -> str:
        """Execute the booking cancellation."""
        result = self.api_client.cancel_booking(booking_id)
        
        if result['success']:
            return f"""✅ Booking {booking_id} has been cancelled successfully.

We're sorry to see you go! Feel free to make a new reservation anytime."""
        else:
            return f"❌ Failed to cancel booking: {result.get('error', 'Unknown error')}"