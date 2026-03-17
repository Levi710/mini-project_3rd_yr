import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in .env")
        return

    print(f"Testing Gemini API key: {api_key[:10]}...")
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Hello"],
        )
        print("Gemini API test SUCCESSFUL")
        print("Response:", response.text)
    except Exception as e:
        print("Gemini API test FAILED")
        print("Error:", str(e))

if __name__ == "__main__":
    test_gemini()
