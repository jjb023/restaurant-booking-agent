#!/usr/bin/env python3
"""Test script for greeting handling."""

import os
import sys
from agent_ollama import BookingAgent
from booking_client import BookingAPIClient

def test_greetings():
    """Test that greetings are handled correctly without tool calls."""
    
    # Initialize the API client
    base_url = os.getenv('BOOKING_API_URL', 'http://localhost:8547')
    bearer_token = os.getenv('BOOKING_API_TOKEN', 'test-token')
    
    api_client = BookingAPIClient(base_url, bearer_token)
    agent = BookingAgent(api_client)
    
    # Test conversation flow
    session_id = "test_greeting_001"
    
    print("🧪 Testing Greeting Handling\n")
    print("=" * 50)
    
    # Test simple greetings
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
    
    for greeting in greetings:
        print(f"\nUser: {greeting}")
        response = agent.process_message(greeting, session_id)
        print(f"Agent: {response}")
        print("-" * 30)
    
    # Test booking request
    print(f"\nUser: josh beal, 7pm, 4 people, on sunday")
    response = agent.process_message("josh beal, 7pm, 4 people, on sunday", session_id)
    print(f"Agent: {response}")
    
    print("\n" + "=" * 50)
    print("✅ Greeting test completed!")

if __name__ == "__main__":
    print("🚀 Starting Greeting Tests")
    
    try:
        test_greetings()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
