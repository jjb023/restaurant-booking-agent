"""Test script to verify API authentication and get the correct token."""

import requests
from urllib.parse import urlencode

# Test different authentication approaches
base_url = "http://localhost:8547"

# The token from the documentation
doc_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8"

# Common test tokens
test_tokens = [
    doc_token,
    "test_token",
    "Bearer test_token",
    ""  # No token
]

print("Testing API Authentication...")
print("=" * 50)

for i, token in enumerate(test_tokens, 1):
    print(f"\nTest {i}: Token = {token[:20]}..." if len(token) > 20 else f"\nTest {i}: Token = '{token}'")
    
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}' if not token.startswith('Bearer') else token
    headers['Content-Type'] = 'application/x-www-form-urlencoded'
    
    data = {
        'VisitDate': '2025-08-10',
        'PartySize': '2',
        'ChannelCode': 'ONLINE'
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/ConsumerApi/v1/Restaurant/TheHungryUnicorn/AvailabilitySearch",
            data=urlencode(data),
            headers=headers
        )
        
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  ✅ SUCCESS! This token works.")
            print(f"  Response: {response.json()}")
            print(f"\n  Use this token in your .env file:")
            print(f"  BOOKING_API_TOKEN={token}")
            break
        else:
            print(f"  ❌ Failed: {response.text[:100]}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 50)
print("\nAlternative: Test without authentication requirement")
print("Trying without any authentication...")

try:
    response = requests.post(
        f"{base_url}/api/ConsumerApi/v1/Restaurant/TheHungryUnicorn/AvailabilitySearch",
        data=urlencode(data)
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ API works without authentication!")
        print("You can set BOOKING_API_TOKEN to an empty string or 'test_token'")
except Exception as e:
    print(f"Error: {e}")