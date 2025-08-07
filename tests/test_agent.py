"""Test suite for the booking agent."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from booking_client import BookingAPIClient
from tools import DateTimeParser, CheckAvailabilityTool, CreateBookingTool
from agent import BookingAgent


class TestDateTimeParser:
    """Test the date and time parsing utilities."""
    
    def test_parse_today(self):
        today = datetime.now().strftime('%Y-%m-%d')
        assert DateTimeParser.parse_date('today') == today
        assert DateTimeParser.parse_date('Today') == today
    
    def test_parse_tomorrow(self):
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        assert DateTimeParser.parse_date('tomorrow') == tomorrow
    
    def test_parse_weekend(self):
        # Should return next Saturday
        today = datetime.now()
        days_ahead = 5 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        weekend = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        assert DateTimeParser.parse_date('this weekend') == weekend
    
    def test_parse_time_with_pm(self):
        assert DateTimeParser.parse_time('7pm') == '19:00'
        assert DateTimeParser.parse_time('7:30pm') == '19:30'
        assert DateTimeParser.parse_time('12pm') == '12:00'
    
    def test_parse_time_with_am(self):
        assert DateTimeParser.parse_time('9am') == '09:00'
        assert DateTimeParser.parse_time('12am') == '00:00'
    
    def test_parse_time_without_meridiem(self):
        # Should assume evening for dinner times
        assert DateTimeParser.parse_time('7:00') == '19:00'
        assert DateTimeParser.parse_time('8') == '20:00'


class TestBookingAPIClient:
    """Test the booking API client."""
    
    @patch('requests.get')
    def test_check_availability_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'available_slots': ['18:00', '19:00', '20:00']
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = BookingAPIClient('http://localhost:8547', 'test_token')
        result = client.check_availability('2024-03-15', '19:00', 4)
        
        assert result['success'] is True
        assert 'available_slots' in result['data']
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_check_availability_error(self, mock_get):
        mock_get.side_effect = Exception('Connection error')
        
        client = BookingAPIClient('http://localhost:8547', 'test_token')
        result = client.check_availability('2024-03-15')
        
        assert result['success'] is False
        assert 'error' in result
    
    @patch('requests.post')
    def test_create_booking_success(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            'booking_id': 'BK123456',
            'status': 'confirmed'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = BookingAPIClient('http://localhost:8547', 'test_token')
        result = client.create_booking(
            'John Doe', '2024-03-15', '19:00', 4, '555-1234'
        )
        
        assert result['success'] is True
        assert result['data']['booking_id'] == 'BK123456'
    
    @patch('requests.delete')
    def test_cancel_booking_success(self, mock_delete):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_delete.return_value = mock_response
        
        client = BookingAPIClient('http://localhost:8547', 'test_token')
        result = client.cancel_booking('BK123456')
        
        assert result['success'] is True
        assert 'cancelled successfully' in result['message']


class TestBookingTools:
    """Test the LangChain tools."""
    
    def test_check_availability_tool(self):
        mock_client = Mock()
        mock_client.check_availability.return_value = {
            'success': True,
            'data': {'available_slots': ['18:00', '19:00', '20:00']}
        }
        
        tool = CheckAvailabilityTool(mock_client)
        result = tool._run('next Friday', '7pm', 4)
        
        assert 'Available time slots' in result
        mock_client.check_availability.assert_called_once()
    
    def test_create_booking_tool(self):
        mock_client = Mock()
        mock_client.create_booking.return_value = {
            'success': True,
            'data': {'booking_id': 'BK123456'}
        }
        
        tool = CreateBookingTool(mock_client)
        result = tool._run(
            'John Doe', 'tomorrow', '7pm', 4, '555-1234', 'Window seat'
        )
        
        assert 'Booking confirmed' in result
        assert 'BK123456' in result


class TestBookingAgent:
    """Test the booking agent."""
    
    @patch('langchain_openai.ChatOpenAI')
    def test_agent_initialization(self, mock_llm):
        mock_client = Mock()
        agent = BookingAgent(mock_client)
        
        assert agent.api_client == mock_client
        assert len(agent.tools) == 5
        assert agent.session_data == {}
    
    @patch('langchain_openai.ChatOpenAI')
    def test_process_message(self, mock_llm):
        mock_client = Mock()
        agent = BookingAgent(mock_client)
        
        # Mock the agent executor
        agent.agent_executor.run = Mock(return_value="I can help you with that!")
        
        response = agent.process_message("Check availability for tomorrow")
        
        assert response == "I can help you with that!"
        agent.agent_executor.run.assert_called_once()
    
    @patch('langchain_openai.ChatOpenAI')
    def test_clear_memory(self, mock_llm):
        mock_client = Mock()
        agent = BookingAgent(mock_client)
        
        # Add some session data
        session_id = 'test_session'
        agent.session_data[session_id] = {'test': 'data'}
        
        # Clear memory
        agent.clear_memory(session_id)
        
        assert session_id not in agent.session_data
