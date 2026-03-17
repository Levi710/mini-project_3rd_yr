import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found in .env")
        return

    print(f"Testing Groq API key: {api_key[:10]}...")
    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        print("Groq API test SUCCESSFUL")
        print("Response:", chat_completion.choices[0].message.content)
    except Exception as e:
        print("Groq API test FAILED")
        print("Error:", str(e))

if __name__ == "__main__":
    test_groq()
