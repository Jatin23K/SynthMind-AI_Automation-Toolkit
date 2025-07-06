import os

from dotenv import load_dotenv


def load_config():
    """Load configuration settings from environment variables.

    For production, it's recommended to set these environment variables directly
    in your deployment environment (e.g., Docker, Kubernetes, cloud service config)
    rather than relying solely on a .env file.
    """
    # Load environment variables from .env file (for local development convenience)
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

    config = {
        "llm_provider": os.getenv("LLM_PROVIDER", "openai").lower(), # Default to openai
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"), # Default LLM model
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)), # Default LLM temperature
        "log_level": os.getenv("LOG_LEVEL", "INFO").upper(), # Default logging level
        "output_file_format": os.getenv("OUTPUT_FILE_FORMAT", "csv").lower(), # Default output format
        # Add other configuration variables here as needed
    }

    # Validate API keys based on provider
    if config["llm_provider"] == "openai" and not config["openai_api_key"]:
        raise EnvironmentError("OPENAI_API_KEY not set for OpenAI provider.")
    elif config["llm_provider"] == "google" and not config["google_api_key"]:
        raise EnvironmentError("GOOGLE_API_KEY not set for Google provider.")
    elif config["llm_provider"] == "anthropic" and not config["anthropic_api_key"]:
        raise EnvironmentError("ANTHROPIC_API_KEY not set for Anthropic provider.")

    return config
