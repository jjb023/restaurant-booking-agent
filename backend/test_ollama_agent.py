#!/usr/bin/env python3
"""Test script for the improved Ollama agent."""

import os
import sys
from agent_ollama import BookingAgent
from booking_client import BookingAPIClient

def test_ollama_agent():
    """Test the Ollama agent with natural language understanding."""
    
    # Initialize the API client
    base_url = os.getenv('BOOKING_API_URL', 'http://localhost:8547')
    bearer_token = os.getenv('BOOKING_API_TOKEN', 'test-token')
    
    api_client = BookingAPIClient(base_url, bearer_token)
    agent = BookingAgent(api_client)
    
    # Test conversation flow
    session_id = "test_session_001"
    
    print("🧪 Testing Improved Ollama Agent\n")
    print("=" * 50)
    
    # Test 1: User provides all details in one message
    print("\n1️⃣ User: josh beal, 7pm, 4 people, on sunday")
    response = agent.process_message("josh beal, 7pm, 4 people, on sunday", session_id)
    print(f"Agent: {response}")
    
    # Test 2: Check availability
    print("\n2️⃣ User: Check availability for this weekend")
    response = agent.process_message("Check availability for this weekend", session_id)
    print(f"Agent: {response}")
    
    # Test 3: Book with details
    print("\n3️⃣ User: Book a table for John Smith tomorrow at 8pm for 6 people")
    response = agent.process_message("Book a table for John Smith tomorrow at 8pm for 6 people", session_id)
    print(f"Agent: {response}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")

def test_natural_language():
    """Test natural language understanding."""
    
    api_client = BookingAPIClient("http://localhost:8547", "test-token")
    agent = BookingAgent(api_client)
    
    print("\n🧪 Testing Natural Language Understanding\n")
    print("=" * 50)
    
    test_messages = [
        "Hi, I'd like to book a table for 4 people on Saturday at 7pm under the name Sarah Johnson",
        "Can you check if you have availability for tomorrow evening?",
        "I want to make a reservation for 6 people, my name is Mike, we'd like to come on Friday at 8pm",
        "Book me a table for 2 people tonight at 6pm, my name is Lisa"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}️⃣ User: {message}")
        response = agent.process_message(message, f"test_session_{i}")
        print(f"Agent: {response}")
    
    print("\n" + "=" * 50)
    print("✅ Natural language test completed!")

if __name__ == "__main__":
    print("🚀 Starting Ollama Agent Tests")
    
    try:
        test_ollama_agent()
        test_natural_language()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
