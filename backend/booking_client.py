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
            # Correct endpoint path
            endpoint = f"{self.base_url}/api/ConsumerApi/v1/Restaurant/{self.restaurant_name}/AvailabilitySearch"
            
            # Prepare form data
            data = {
                'VisitDate': date,
                'PartySize': str(party_size) if party_size else '2',
                'ChannelCode': 'ONLINE'
            }
            
            response = requests.post(
                endpoint,
                data=urlencode(data),
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
            endpoint = f"{self.base_url}/api/ConsumerApi/v1/Restaurant/{self.restaurant_name}/BookingWithStripeToken"
            
            # Parse name into first and last
            name_parts = customer_name.strip().split(' ', 1)
            first_name = name_parts[0]
            surname = name_parts[1] if len(name_parts) > 1 else ''
            
            # Ensure time has seconds
            if len(time.split(':')) == 2:
                time = f"{time}:00"
            
            data = {
                'VisitDate': date,
                'VisitTime': time,
                'PartySize': str(party_size),
                'ChannelCode': 'ONLINE',
                'Customer[FirstName]': first_name,
                'Customer[Surname]': surname
            }
            
            if contact_number:
                data['Customer[Mobile]'] = contact_number
            if special_requests:
                data['SpecialRequests'] = special_requests
            
            response = requests.post(
                endpoint,
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
            # Correct endpoint path
            endpoint = f"{self.base_url}/api/ConsumerApi/v1/Restaurant/{self.restaurant_name}/Booking/{booking_id}"
            
            response = requests.get(
                endpoint,
                headers={'Authorization': f'Bearer {self.bearer_token}'}
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
            endpoint = f"{self.base_url}/api/ConsumerApi/v1/Restaurant/{self.restaurant_name}/Booking/{booking_id}"
            
            data = {}
            if 'date' in kwargs:
                data['VisitDate'] = kwargs['date']
            if 'time' in kwargs:
                time = kwargs['time']
                if len(time.split(':')) == 2:
                    time = f"{time}:00"
                data['VisitTime'] = time
            if 'party_size' in kwargs:
                data['PartySize'] = str(kwargs['party_size'])
            if 'special_requests' in kwargs:
                data['SpecialRequests'] = kwargs['special_requests']
            
            response = requests.patch(
                endpoint,
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
            endpoint = f"{self.base_url}/api/ConsumerApi/v1/Restaurant/{self.restaurant_name}/Booking/{booking_id}/Cancel"
            
            data = {
                'micrositeName': self.restaurant_name,
                'bookingReference': booking_id,
                'cancellationReasonId': '1'  
            }
            
            response = requests.post(
                endpoint,
                data=urlencode(data),
                headers=self.headers
            )
            response.raise_for_status()
            return {'success': True, 'message': 'Booking cancelled successfully'}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error cancelling booking: {e}")
            return {'success': False, 'error': str(e)}