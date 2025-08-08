"""Simple date and time parsing utilities."""

from datetime import datetime, timedelta
import re


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
        elif 'weekend' in date_str or 'saturday' in date_str:
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
        
        return date_str
    
    @staticmethod
    def parse_time(time_str: str) -> str:
        """Convert natural language time to HH:MM format."""
        if not time_str:
            return '19:00'
            
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
                if hour <= 5:
                    hour += 12
            
            return f"{hour:02d}:{minute:02d}"
        
        return time_str