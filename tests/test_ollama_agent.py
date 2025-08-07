"""test_ollama_agent.py - Test the Ollama integration"""

import requests
import json

def test_ollama_agent():
    base_url = "http://localhost:8000"
    
    # Test health check
    response = requests.get(f"{base_url}/health")
    print(f"Health Check: {response.json()}")
    
    # Test conversations
    test_messages = [
        "Hi, I'd like to make a reservation",
        "Show me availability for this weekend",
        "Book a table for 4 people tomorrow at 7pm",
        "My name is John Smith",
        "What's my booking reference?"
    ]
    
    session_id = None
    for message in test_messages:
        response = requests.post(
            f"{base_url}/chat",
            json={"message": message, "session_id": session_id}
        )
        data = response.json()
        session_id = data.get('session_id')
        print(f"\nUser: {message}")
        print(f"Agent: {data.get('response')}")
        print(f"Model: {data.get('model')}")

if __name__ == "__main__":
    test_ollama_agent()