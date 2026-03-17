import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in .env")
        return

    print(f"Testing OpenAI API key: {api_key[:10]}...")
    
    # We use requests to avoid needing the 'openai' package for a simple check
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            print("OpenAI API test SUCCESSFUL")
            print("Response:", response.json()["choices"][0]["message"]["content"])
        else:
            print(f"OpenAI API test FAILED (Status {response.status_code})")
            print("Error:", response.text)
    except Exception as e:
        print("OpenAI API test ERROR")
        print("Error:", str(e))

if __name__ == "__main__":
    test_openai()
