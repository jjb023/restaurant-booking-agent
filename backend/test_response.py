#!/usr/bin/env python3
"""Test script for response handling."""

import os
import sys
from agent_ollama import BookingAgent
from booking_client import BookingAPIClient

def test_response_handling():
    """Test that responses are properly returned."""
    
    # Initialize the API client
    base_url = os.getenv('BOOKING_API_URL', 'http://localhost:8547')
    bearer_token = os.getenv('BOOKING_API_TOKEN', 'test-token')
    
    api_client = BookingAPIClient(base_url, bearer_token)
    agent = BookingAgent(api_client)
    
    # Test conversation flow
    session_id = "test_response_001"
    
    print("🧪 Testing Response Handling\n")
    print("=" * 50)
    
    # Test availability request
    print("\n1️⃣ User: check availability for saturday")
    response = agent.process_message("check availability for saturday", session_id)
    print(f"Agent Response: {response}")
    print(f"Response Length: {len(response)}")
    print("-" * 50)
    
    # Test booking request
    print("\n2️⃣ User: josh beal, 7pm, 4 people, on sunday")
    response = agent.process_message("josh beal, 7pm, 4 people, on sunday", session_id)
    print(f"Agent Response: {response}")
    print(f"Response Length: {len(response)}")
    print("-" * 50)
    
    # Test greeting
    print("\n3️⃣ User: hello")
    response = agent.process_message("hello", session_id)
    print(f"Agent Response: {response}")
    print(f"Response Length: {len(response)}")
    
    print("\n" + "=" * 50)
    print("✅ Response test completed!")

if __name__ == "__main__":
    print("🚀 Starting Response Tests")
    
    try:
        test_response_handling()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
