"""
Test script to verify Gemini API connection
"""
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test Gemini API connection with a simple query"""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env file")
        return False

    try:
        # Create a client
        client = genai.Client(api_key=api_key)

        # List available models first
        print("Listing available models...")
        for model in client.models.list():
            if 'generateContent' in model.supported_actions:
                print(f"  - {model.name}")

        # Test with a simple query using a common model
        print("\nTesting API with models/gemini-2.0-flash...")
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents="What is machine learning? Answer in one sentence."
        )

        print("\nSUCCESS: Gemini API is working!")
        print(f"\nTest query: 'What is machine learning? Answer in one sentence.'")
        print(f"Response: {response.text}")

        return True

    except Exception as e:
        print(f"ERROR: Failed to connect to Gemini API")
        print(f"Details: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api()
