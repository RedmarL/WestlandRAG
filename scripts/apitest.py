import os

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"✅ Found API key: starts with {api_key[:5]}... and ends with ...{api_key[-4:]}")
else:
    print("❌ No OPENAI_API_KEY found in environment variables.")
