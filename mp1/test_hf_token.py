import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_hf():
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("HUGGINGFACE_TOKEN not found in environment")
        return

    print(f"Testing Hugging Face Token: {token[:10]}...")
    
    # Use the standard Inference API path
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    
    payload = {
        "inputs": "Hello",
        "parameters": {"max_new_tokens": 10}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            print("Hugging Face API test SUCCESSFUL")
            print("Response:", response.json())
        else:
            print(f"Hugging Face API test FAILED (Status {response.status_code})")
            print("Error:", response.text)
    except Exception as e:
        print("Hugging Face API test ERROR")
        print("Error:", str(e))

if __name__ == "__main__":
    test_hf()
