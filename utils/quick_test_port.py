# quick_test.py
import requests
import json

url = "http://localhost:1234/v1/chat/completions"
payload = {
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.1,
    "max_tokens": 50,
    "stream": False
}

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Connection error: {e}")