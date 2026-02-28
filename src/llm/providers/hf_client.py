import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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


class HFLLMClient(LLMClient):
    """HFLLMClient is an implementation of LLMClient that uses Hugging Face transformers to generate text.

    Args:
        LLMClient: The base LLMClient class that defines the interface for generating text.
    """
    def __init__(self, model_name: str,
                *,
                torch_dtype=torch.float16,
                device_map: str = "auto",
                default_temperature: float = 0.2,
                default_max_new_tokens: int = 400,
                default_do_sample: bool = False):


        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self.default_do_sample = default_do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)

    @staticmethod
    def _extract_json_payload(text: str) -> object | None:
        """Best-effort JSON extraction from model output text."""
        stripped = text.strip()
        if not stripped:
            return None

        candidates: list[str] = [stripped]
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
        candidates.extend(fenced)

        decoder = json.JSONDecoder()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

            # Fallback: find first decodable JSON object/array within the string.
            for idx, ch in enumerate(candidate):
                if ch not in "{[":
                    continue
                try:
                    obj, _ = decoder.raw_decode(candidate[idx:])
                    return obj
                except json.JSONDecodeError:
                    continue

        return None

    @staticmethod
    def _normalize_tool_calls(payload: object) -> list[dict[str, object]] | None:
        """Convert model-emitted tool call JSON into the normalized internal schema."""
        if not isinstance(payload, dict):
            return None

        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list):
            return None

        normalized: list[dict[str, object]] = []
        for idx, call in enumerate(raw_calls):
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            if not isinstance(fn, dict):
                continue

            name = str(fn.get("name", "")).strip()
            if not name:
                continue

            args = fn.get("arguments", {})
            if isinstance(args, str):
                args_json = args
            else:
                args_json = json.dumps(args)

            normalized.append(
                {
                    "id": str(call.get("id", f"call_{idx+1}")),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_json,
                    },
                }
            )

        return normalized or None

    @staticmethod
    def _prepare_messages_for_template(messages: list[dict[str, object]]) -> list[dict[str, str]]:
        """
        Normalize agent-level messages into a conservative subset supported by
        most HF chat templates (system/user/assistant with content only).
        """
        normalized: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", "") or "")

            if role == "tool":
                tool_name = str(msg.get("name", "tool"))
                normalized.append(
                    {
                        "role": "user",
                        "content": f"[Tool result: {tool_name}] {content}",
                    }
                )
                continue

            if role == "assistant" and msg.get("tool_calls"):
                tool_calls_text = json.dumps(msg.get("tool_calls"), ensure_ascii=False)
                combined = (content + "\n" + f"[Tool calls] {tool_calls_text}").strip()
                normalized.append({"role": "assistant", "content": combined})
                continue

            if role not in {"system", "user", "assistant"}:
                role = "user"

            normalized.append({"role": role, "content": content})

        return normalized
        
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
        
        _ = json_mode  # Reserved for future JSON-constrained decoding support.
        
        temperature = self.default_temperature if temperature is None else temperature
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        do_sample = self.default_do_sample
        if stage == "glossary":
            # Prefer deterministic decoding for parseable tool JSON.
            do_sample = False
            temperature = 0.0

        if messages is None:
            if prompt is None:
                raise ValueError("Either messages or prompt must be provided.")
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages = self._prepare_messages_for_template(messages)

        tools = [GOOGLE_SEARCH_SCHEMA] if stage == "glossary" else None
        
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            
        except TypeError:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        
        outputs = self.model.generate(input_ids=input_ids, 
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    do_sample=do_sample)
        

        #count tokens for usage stats..
        prompt_len = input_ids.shape[1]
        total_len = outputs.shape[1]
        usage = LLMUsage(
            prompt_tokens=prompt_len,
            completion_tokens=total_len - prompt_len,
            total_tokens=total_len)
        
        decoded_text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
        response_text = decoded_text
        parsed_payload = self._extract_json_payload(response_text) if stage == "glossary" else None
        tool_calls = self._normalize_tool_calls(parsed_payload) if stage == "glossary" else None

        # When model emitted tool calls, agent should execute tools and continue the loop.
        if tool_calls:
            response_text = ""

        return LLMResult(
            text=response_text,
            model=self.model.config._name_or_path,
            usage=usage,
            tool_calls=tool_calls,
            raw={"decoded_text": decoded_text},
        )

        
