"""API client for the restaurant booking server."""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import logging
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class BookingAPIClient:
    """Client for interacting with the restaurant booking API."""
    
    def __init__(self, base_url: str, bearer_token: str, restaurant_name: str = "TheHungryUnicorn"):
        self.base_url = base_url.rstrip('/')
        self.bearer_token = bearer_token
        self.restaurant_name = restaurant_name
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    
    def check_availability(self, date: str, time: Optional[str] = None, party_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Check availability for a specific date and optionally time.
        
        Args:
            date: Date in YYYY-MM-DD format
            time: Optional time in HH:MM format
            party_size: Optional number of people
        """
        try:
            params = {'restaurant': self.restaurant_name, 'date': date}
            if time:
                params['time'] = time
            if party_size:
                params['party_size'] = party_size
            
            response = requests.get(
                f"{self.base_url}/availability",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return {'success': True, 'data': response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking availability: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_booking(self, customer_name: str, date: str, time: str, party_size: int, 
                      contact_number: Optional[str] = None, special_requests: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new booking.
        
        Args:
            customer_name: Name of the customer
            date: Date in YYYY-MM-DD format
            time: Time in HH:MM format
            party_size: Number of people
            contact_number: Optional contact number
            special_requests: Optional special requests
        """
        try:
            data = {
                'restaurant': self.restaurant_name,
                'customer_name': customer_name,
                'date': date,
                'time': time,
                'party_size': str(party_size)
            }
            
            if contact_number:
                data['contact_number'] = contact_number
            if special_requests:
                data['special_requests'] = special_requests
            
            response = requests.post(
                f"{self.base_url}/bookings",
                data=urlencode(data),
                headers=self.headers
            )
            response.raise_for_status()
            return {'success': True, 'data': response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating booking: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_booking(self, booking_id: str) -> Dict[str, Any]:
        """
        Retrieve booking details.
        
        Args:
            booking_id: The booking reference ID
        """
        try:
            response = requests.get(
                f"{self.base_url}/bookings/{booking_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return {'success': True, 'data': response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving booking: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_booking(self, booking_id: str, **kwargs) -> Dict[str, Any]:
        """
        Update an existing booking.
        
        Args:
            booking_id: The booking reference ID
            **kwargs: Fields to update (date, time, party_size, etc.)
        """
        try:
            data = {k: v for k, v in kwargs.items() if v is not None}
            
            response = requests.put(
                f"{self.base_url}/bookings/{booking_id}",
                data=urlencode(data),
                headers=self.headers
            )
            response.raise_for_status()
            return {'success': True, 'data': response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error updating booking: {e}")
            return {'success': False, 'error': str(e)}
    
    def cancel_booking(self, booking_id: str) -> Dict[str, Any]:
        """
        Cancel a booking.
        
        Args:
            booking_id: The booking reference ID
        """
        try:
            response = requests.delete(
                f"{self.base_url}/bookings/{booking_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return {'success': True, 'message': 'Booking cancelled successfully'}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error cancelling booking: {e}")
            return {'success': False, 'error': str(e)}