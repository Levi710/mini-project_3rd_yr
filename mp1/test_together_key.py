import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_together():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("TOGETHER_API_KEY not found in .env")
        return

    print(f"Testing Together AI API key: {api_key[:10]}...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            print("Together AI API test SUCCESSFUL")
            print("Response:", response.json()["choices"][0]["message"]["content"])
        else:
            print(f"Together AI API test FAILED (Status {response.status_code})")
            print("Error:", response.text)
    except Exception as e:
        print("Together AI API test ERROR")
        print("Error:", str(e))

if __name__ == "__main__":
    test_together()
