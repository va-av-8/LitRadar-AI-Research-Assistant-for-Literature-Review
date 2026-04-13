"""OpenRouter LLM client with Langfuse tracing."""

import json
from typing import Any, Optional, Dict
from openai import OpenAI

from .config import get_settings
from .logger import logger


class LLMClient:
    """OpenRouter-compatible LLM client with observability."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.settings.openrouter_api_key,
        )
        self.langfuse = None
        self._root_spans: Dict[str, Any] = {}  # session_id -> root span
        self._propagate_ctx = None

        if self.settings.langfuse_tracing and self.settings.langfuse_public_key:
            try:
                from langfuse import get_client
                self.langfuse = get_client()
                logger.info("langfuse_initialized")
            except Exception as e:
                logger.warning("langfuse_init_failed", error=str(e))

    def start_trace(self, session_id: str, query: str = "") -> Any:
        """Start a new trace for session."""
        if not self.langfuse or not session_id:
            return None

        if session_id not in self._root_spans:
            # Import propagate_attributes for session tracking
            from langfuse import propagate_attributes

            # Set session_id for all child observations
            self._propagate_ctx = propagate_attributes(session_id=session_id)
            self._propagate_ctx.__enter__()

            # Create root span for the session
            root_span = self.langfuse.start_observation(
                name=f"session:{session_id}",
                input={"query": query} if query else None,
            )
            self._root_spans[session_id] = root_span
        return self._root_spans[session_id]

    def end_trace(self, session_id: str, output: dict = None):
        """End and flush trace for session."""
        if not self.langfuse or session_id not in self._root_spans:
            return

        root_span = self._root_spans[session_id]
        if output:
            root_span.update(output=output)
        root_span.end()

        # Exit propagate_attributes context
        if hasattr(self, '_propagate_ctx') and self._propagate_ctx:
            self._propagate_ctx.__exit__(None, None, None)
            self._propagate_ctx = None

        # Flush to ensure data is sent
        self.langfuse.flush()
        del self._root_spans[session_id]

    def get_root_span(self, session_id: str) -> Any:
        """Get root span for session."""
        return self._root_spans.get(session_id)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
        session_id: str = "",
        agent_name: str = "",
    ) -> tuple[str, int, int]:
        """
        Call LLM and return (response_text, input_tokens, output_tokens).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (defaults to LLM_MODEL_DEFAULT)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_format: Optional format spec (e.g., {"type": "json_object"})
            session_id: For logging
            agent_name: For logging

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        model = model or self.settings.llm_model_default

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        # Get root span for Langfuse generation
        root_span = self.get_root_span(session_id)
        generation = None

        if root_span:
            generation = root_span.start_observation(
                name=f"{agent_name}_llm_call",
                as_type="generation",
                model=model,
                input=messages,
            )

        try:
            response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            # End Langfuse generation with output
            if generation:
                generation.update(
                    output=content,
                    usage_details={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )
                generation.end()

            logger.info(
                "llm_call",
                session_id=session_id,
                agent_name=agent_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            return content, input_tokens, output_tokens

        except Exception as e:
            # End Langfuse generation with error
            if generation:
                generation.update(
                    output=None,
                    metadata={"error": str(e)},
                )
                generation.end()

            logger.error(
                "llm_call_failed",
                session_id=session_id,
                agent_name=agent_name,
                error=str(e),
            )
            raise

    def chat_completion_json(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        session_id: str = "",
        agent_name: str = "",
        retries: int = 2,
    ) -> tuple[dict[str, Any], int, int]:
        """
        Call LLM expecting JSON response, with retry on parse errors.

        Returns:
            Tuple of (parsed_json, total_input_tokens, total_output_tokens)
        """
        total_input = 0
        total_output = 0

        for attempt in range(retries + 1):
            try:
                content, input_tokens, output_tokens = self.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    session_id=session_id,
                    agent_name=agent_name,
                )
                total_input += input_tokens
                total_output += output_tokens

                # Parse JSON
                parsed = json.loads(content)
                return parsed, total_input, total_output

            except json.JSONDecodeError as e:
                logger.warning(
                    "json_parse_error",
                    session_id=session_id,
                    agent_name=agent_name,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt == retries:
                    raise

        # Should not reach here, but just in case
        raise RuntimeError("Exhausted retries for JSON parsing")


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
