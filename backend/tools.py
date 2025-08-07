"""Custom LangChain tools for restaurant booking operations."""

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
        date_str = date_str.lower()
        today = datetime.now()
        
        # Handle relative dates
        if 'today' in date_str:
            return today.strftime('%Y-%m-%d')
        elif 'tomorrow' in date_str:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'weekend' in date_str:
            # Next Saturday
            days_ahead = 5 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        elif 'friday' in date_str:
            days_ahead = 4 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            if 'next' in date_str:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        # Try to parse standard date formats
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return date_str  # Return as-is if parsing fails
    
    @staticmethod
    def parse_time(time_str: str) -> str:
        """Convert natural language time to HH:MM format."""
        time_str = time_str.lower().replace('.', '')
        
        # Handle am/pm format
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            meridiem = time_match.group(3)
            
            if meridiem == 'pm' and hour < 12:
                hour += 12
            elif meridiem == 'am' and hour == 12:
                hour = 0
            elif not meridiem and hour < 12 and hour >= 6:
                # Assume evening for dinner times
                hour += 12
            
            return f"{hour:02d}:{minute:02d}"
        
        return time_str


class CheckAvailabilityInput(BaseModel):
    date: str = Field(description="Date to check availability (e.g., 'next Friday', '2024-03-15')")
    time: Optional[str] = Field(None, description="Optional time to check (e.g., '7pm', '19:00')")
    party_size: Optional[int] = Field(None, description="Number of people")


class CheckAvailabilityTool(BaseTool):
    name = "check_availability"
    description = "Check restaurant availability for a specific date and optionally time"
    args_schema: Type[BaseModel] = CheckAvailabilityInput
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, date: str, time: Optional[str] = None, party_size: Optional[int] = None) -> str:
        parsed_date = DateTimeParser.parse_date(date)
        parsed_time = DateTimeParser.parse_time(time) if time else None
        
        result = self.api_client.check_availability(parsed_date, parsed_time, party_size)
        
        if result['success']:
            data = result['data']
            if 'available_slots' in data:
                slots = data['available_slots']
                if slots:
                    return f"Available time slots for {parsed_date}: {', '.join(slots)}"
                else:
                    return f"No available slots for {parsed_date}"
            return f"Restaurant is available on {parsed_date}" + (f" at {parsed_time}" if parsed_time else "")
        else:
            return f"Error checking availability: {result['error']}"


class CreateBookingInput(BaseModel):
    customer_name: str = Field(description="Name of the customer")
    date: str = Field(description="Date for the booking")
    time: str = Field(description="Time for the booking")
    party_size: int = Field(description="Number of people")
    contact_number: Optional[str] = Field(None, description="Contact phone number")
    special_requests: Optional[str] = Field(None, description="Any special requests")


class CreateBookingTool(BaseTool):
    name = "create_booking"
    description = "Create a new restaurant booking"
    args_schema: Type[BaseModel] = CreateBookingInput
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, customer_name: str, date: str, time: str, party_size: int,
             contact_number: Optional[str] = None, special_requests: Optional[str] = None) -> str:
        parsed_date = DateTimeParser.parse_date(date)
        parsed_time = DateTimeParser.parse_time(time)
        
        result = self.api_client.create_booking(
            customer_name, parsed_date, parsed_time, party_size,
            contact_number, special_requests
        )
        
        if result['success']:
            booking = result['data']
            return f"Booking confirmed! Reference: {booking.get('booking_id', 'N/A')}. " \
                   f"Table for {party_size} on {parsed_date} at {parsed_time}"
        else:
            return f"Failed to create booking: {result['error']}"


class GetBookingInput(BaseModel):
    booking_id: str = Field(description="The booking reference ID")


class GetBookingTool(BaseTool):
    name = "get_booking"
    description = "Retrieve details of an existing booking"
    args_schema: Type[BaseModel] = GetBookingInput
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, booking_id: str) -> str:
        result = self.api_client.get_booking(booking_id)
        
        if result['success']:
            booking = result['data']
            return f"Booking Details:\n" \
                   f"Reference: {booking.get('booking_id')}\n" \
                   f"Name: {booking.get('customer_name')}\n" \
                   f"Date: {booking.get('date')}\n" \
                   f"Time: {booking.get('time')}\n" \
                   f"Party Size: {booking.get('party_size')}\n" \
                   f"Status: {booking.get('status', 'Confirmed')}"
        else:
            return f"Could not retrieve booking: {result['error']}"


class UpdateBookingInput(BaseModel):
    booking_id: str = Field(description="The booking reference ID")
    date: Optional[str] = Field(None, description="New date for the booking")
    time: Optional[str] = Field(None, description="New time for the booking")
    party_size: Optional[int] = Field(None, description="New party size")


class UpdateBookingTool(BaseTool):
    name = "update_booking"
    description = "Modify an existing booking (change date, time, or party size)"
    args_schema: Type[BaseModel] = UpdateBookingInput
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, booking_id: str, date: Optional[str] = None, 
             time: Optional[str] = None, party_size: Optional[int] = None) -> str:
        update_data = {}
        if date:
            update_data['date'] = DateTimeParser.parse_date(date)
        if time:
            update_data['time'] = DateTimeParser.parse_time(time)
        if party_size:
            update_data['party_size'] = party_size
        
        if not update_data:
            return "No changes specified for the booking"
        
        result = self.api_client.update_booking(booking_id, **update_data)
        
        if result['success']:
            changes = []
            if date:
                changes.append(f"date to {update_data['date']}")
            if time:
                changes.append(f"time to {update_data['time']}")
            if party_size:
                changes.append(f"party size to {party_size}")
            
            return f"Booking {booking_id} updated successfully. Changed: {', '.join(changes)}"
        else:
            return f"Failed to update booking: {result['error']}"


class CancelBookingInput(BaseModel):
    booking_id: str = Field(description="The booking reference ID to cancel")


class CancelBookingTool(BaseTool):
    name = "cancel_booking"
    description = "Cancel an existing booking"
    args_schema: Type[BaseModel] = CancelBookingInput
    
    def __init__(self, api_client: BookingAPIClient):
        super().__init__()
        self.api_client = api_client
    
    def _run(self, booking_id: str) -> str:
        result = self.api_client.cancel_booking(booking_id)
        
        if result['success']:
            return f"Booking {booking_id} has been cancelled successfully"
        else:
            return f"Failed to cancel booking: {result['error']}"
