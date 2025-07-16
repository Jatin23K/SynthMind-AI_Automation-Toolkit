import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables.")
else:
    try:
        print(f"Attempting to use API key: {api_key[:5]}...{api_key[-5:]}")
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", api_key=api_key)
        response = llm.invoke("Hello, world!")
        print("API call successful!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"API call failed: {e}")
