"""
llm/client.py — Ollama chat completion client.

The interface deliberately mirrors OpenAI's chat completion API.
This means swapping to OpenAI in production = change the base URL
and add an Authorization header. The rest of the call site is identical.

Ollama's /api/chat endpoint:
  POST /api/chat
  {"model": "llama3.2:3b", "messages": [...], "stream": false}
  -> {"message": {"role": "assistant", "content": "..."}, "done": true}
"""
import httpx
from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)


def chat_completion(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, int]:
    """
    Call Ollama's chat endpoint and return (response_text, estimated_token_count).

    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": str}
        temperature: Override the default from settings if provided
        max_tokens: Override the default from settings if provided

    Returns:
        (answer_text, token_count)
    """
    settings = get_settings()

    payload = {
        "model": settings.ollama_llm_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "num_predict": max_tokens if max_tokens is not None else settings.llm_max_tokens,
        },
    }

    url = f"{settings.ollama_base_url}/api/chat"

    # Long timeout — on 8GB CPU-only, 800 tokens can take 30+ seconds.
    with httpx.Client(timeout=180.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    answer = data["message"]["content"]

    # Ollama returns prompt_eval_count and eval_count in the response
    # Use these for accurate token accounting
    token_count = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

    logger.info(
        "llm_completion",
        extra={
            "model": settings.ollama_llm_model,
            "tokens_generated": data.get("eval_count", 0),
            "tokens_prompt": data.get("prompt_eval_count", 0),
        },
    )

    return answer, token_count
