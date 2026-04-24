import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
else:
    try:
        genai.configure(api_key=api_key)
        print(f"Listing models for API Key: {api_key[:10]}...")
        
        models = genai.list_models()
        available_models = [m.name for m in models]
        
        print("\nAvailable Models:")
        for m in available_models:
            print(f" - {m}")
            
        if not available_models:
            print("No models found. Please check if 'Generative Language API' is enabled in your Google Cloud Console.")
            
    except Exception as e:
        print(f"Error accessing API: {e}")
