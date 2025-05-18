# This file handles application-wide configuration, including logging and API key setup.

import os
import logging
from openai import OpenAI

# Configure logging for the application
# Set the logging level for the 'openai' library to WARNING to reduce verbosity
logging.getLogger("openai").setLevel(logging.WARNING)
# Configure basic logging for the application with a specific format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Key setup function
def get_openai_client():
    """Retrieves the OpenAI API key from environment variables and initializes the OpenAI client.

    It checks for the 'OPENAI_API_KEY' environment variable.
    If the key is not found or is the placeholder value, a warning is logged.
    The function ensures the environment variable is set (even with the placeholder)
    before initializing the OpenAI client, as the library automatically uses it.

    Returns:
        OpenAI: An initialized OpenAI client instance.
    """
    """Retrieves the OpenAI API key and initializes the OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-xxxxxxxxxxxxxxxxxxxx": # Added check for placeholder
        # If placeholder is still there or key is missing, warn and keep placeholder
        logging.warning("OPENAI_API_KEY not found in environment or is placeholder. Using placeholder - this will likely fail.")
        api_key = "sk-xxxxxxxxxxxxxxxxxxxx" # Ensure placeholder is used if env var is missing/empty

    # Initialize OpenAI client with the key (either from env or placeholder)
    # OpenAI library automatically picks up OPENAI_API_KEY from environment if set
    # Or you can explicitly pass api_key=api_key if preferred
    # The current OpenAI library handles OPENAI_API_KEY env var automatically.
    # We just need to ensure it's set (even if to a placeholder for the warning).
    if not os.getenv("OPENAI_API_KEY"): # Ensure it's set for the OpenAI library if not already
         os.environ["OPENAI_API_KEY"] = api_key

    return OpenAI()

# Global client instance to be used by other modules
# This client is initialized once when config.py is imported.
client = get_openai_client()

# Optional: Add other configuration variables here if needed later
# Example: DEFAULT_MODEL = "gpt-3.5-turbo"
# Example: OUTPUT_FILE_NAME = "analysis_results.txt"
