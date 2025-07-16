import hashlib
import json
import logging

class LLMCache:
    """
    A simple in-memory cache for LLM responses.
    """
    _cache = {}
    _logger = logging.getLogger(__name__)

    @staticmethod
    def _generate_key(prompt: str, model: str, temperature: float, **kwargs) -> str:
        """Generates a unique hash key for the cache based on prompt and LLM parameters."""
        # Ensure consistent ordering of kwargs for consistent hashing
        sorted_kwargs = sorted(kwargs.items())
        data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "kwargs": sorted_kwargs
        }
        # Use JSON dumps for a stable string representation, then hash
        return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()

    @classmethod
    def get(cls, prompt: str, model: str, temperature: float, **kwargs):
        """Retrieves a response from the cache if available."""
        key = cls._generate_key(prompt, model, temperature, **kwargs)
        if key in cls._cache:
            cls._logger.info(f"Cache hit for key: {key}")
            return cls._cache[key]
        cls._logger.info(f"Cache miss for key: {key}")
        return None

    @classmethod
    def set(cls, prompt: str, model: str, temperature: float, response, **kwargs):
        """Stores a response in the cache."""
        key = cls._generate_key(prompt, model, temperature, **kwargs)
        cls._cache[key] = response
        cls._logger.info(f"Cache set for key: {key}")
