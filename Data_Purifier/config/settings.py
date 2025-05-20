import os
from dotenv import load_dotenv

def load_config():
    """Load configuration settings from environment variables."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Required environment variables
    required_vars = ["OPENAI_API_KEY"]
    
    # Check for missing environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please create a .env file with these variables."
        )

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        # Add other configuration variables here as needed
    }
