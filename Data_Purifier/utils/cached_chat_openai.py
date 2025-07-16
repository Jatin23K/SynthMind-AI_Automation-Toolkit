import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

class CachedChatOpenAI(BaseChatModel):
    """
    A custom ChatOpenAI wrapper that implements in-memory caching for LLM responses.
    It also supports asynchronous calls.
    """
    _cache = {}
    _logger = logging.getLogger(__name__)
    _llm: ChatOpenAI

    def __init__(self, temperature: float = 0.7, model_name: str = "gpt-4", **kwargs: Any):
        super().__init__(**kwargs)
        self._llm = ChatOpenAI(temperature=temperature, model_name=model_name, **kwargs)
        self._logger.info(f"Initialized CachedChatOpenAI with model: {model_name}, temperature: {temperature}")

    @property
    def _llm_type(self) -> str:
        return "cached_openai_chat"

    def _generate_key(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        """
        Generates a unique hash key for the cache based on messages and LLM parameters.
        """
        # Convert messages to a serializable format
        serializable_messages = []
        for msg in messages:
            serializable_messages.append({"type": msg.type, "content": msg.content})

        # Ensure consistent ordering of kwargs for consistent hashing
        sorted_kwargs = sorted(kwargs.items())
        data = {
            "messages": serializable_messages,
            "model": self._llm.model_name,
            "temperature": self._llm.temperature,
            "kwargs": sorted_kwargs
        }
        # Use JSON dumps for a stable string representation, then hash
        return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, callbacks: Any = None, **kwargs: Any) -> ChatResult:
        """
        Synchronous call to the LLM, with caching.
        """
        key = self._generate_key(messages, stop=stop, **kwargs)

        if key in self._cache:
            self._logger.info(f"Cache hit for key: {key}")
            return self._cache[key]

        self._logger.info(f"Cache miss for key: {key}. Calling LLM...")
        result = self._llm.generate(messages, stop=stop, callbacks=callbacks, **kwargs)
        self._cache[key] = result
        self._logger.info(f"Cache set for key: {key}")
        return result

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, callbacks: Any = None, **kwargs: Any) -> ChatResult:
        """
        Asynchronous call to the LLM, with caching.
        """
        key = self._generate_key(messages, stop=stop, **kwargs)

        if key in self._cache:
            self._logger.info(f"Async Cache hit for key: {key}")
            return self._cache[key]

        self._logger.info(f"Async Cache miss for key: {key}. Calling LLM asynchronously...")
        result = await self._llm.agenerate(messages, stop=stop, callbacks=callbacks, **kwargs)
        self._cache[key] = result
        self._logger.info(f"Async Cache set for key: {key}")
        return result

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self._llm._identifying_params

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        return self._llm.get_num_tokens_from_messages(messages)
