from openai import OpenAI
import os
from src.llm.base import LLMClient, LLMResult, LLMUsage

GOOGLE_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "google_search",
        "description": "Search the web to define hard words for a glossary.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The term to search for"},
            },
            "required": ["query"],
        }
    }
}

class OpenAIClient(LLMClient):
    """Open AI client for the LLMClient

    Args:
        LLMClient: The base LLMClient class that defines the interface for generating text.
    """
    def __init__(self, model: str = "gpt-4o",
                *,
                default_temperature: float = 0.2,
                default_max_new_tokens: int = 400):
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
    
    def generate(
        self,
        stage: str,
        prompt: str | None = None,
        *,
        messages: list[dict[str, object]] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:

        _ = stage  # Stage-specific orchestration lives in the agent layer.
        
        temperature = self.default_temperature if temperature is None else temperature
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens

        if messages is None:
            if prompt is None:
                raise ValueError("Either messages or prompt must be provided.")
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        tools = [GOOGLE_SEARCH_SCHEMA] if stage == "glossary" else None
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        
        tool_calls = None
        if stage == "glossary" and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        text = (message.content or "").strip()

        usage = LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        return LLMResult(text=text,
                        model=self.model,
                        usage=usage,
                        tool_calls=tool_calls,
                        raw=response.model_dump())
