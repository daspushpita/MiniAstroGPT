from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.llm.base import LLMClient, LLMResult, LLMUsage

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:  # pragma: no cover - optional dependency path
    Llama = None
    LlamaGrammar = None


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
        },
    },
}


# Generic JSON grammar for constrained output stages (critic/glossary).
JSON_SCHEMA_GBNF = r"""
root ::= ws object ws
object ::= "{" ws members? ws "}"
members ::= pair (ws "," ws pair)*
pair ::= string ws ":" ws value
value ::= string | object | array | number | "true" | "false" | "null"
array ::= "[" ws elements? ws "]"
elements ::= value (ws "," ws value)*
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
number ::= "-"? int frac? exp?
int ::= "0" | [1-9] [0-9]*
frac ::= "." [0-9]+
exp ::= [eE] [+\-]? [0-9]+
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*
"""


@dataclass
class LlamaCppConfig:
    model_path: str
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    n_batch: int = 512
    n_threads: int = 8
    flash_attn: bool = True
    verbose: bool = False
    seed: int = 42
    default_temperature: float = 0.2
    default_max_new_tokens: int = 400
    enable_grammar: bool = True


class LlamaCppClient(LLMClient):
    def __init__(
        self,
        model_path: str,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: int = 4,
        flash_attn: bool = True,
        verbose: bool = False,
        seed: int = 42,
        default_temperature: float = 0.2,
        default_max_new_tokens: int = 400,
        enable_grammar: bool = True,
        device: str | None = None,
    ):
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
            )

        cfg = LlamaCppConfig(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            flash_attn=flash_attn,
            verbose=verbose,
            seed=seed,
            default_temperature=default_temperature,
            default_max_new_tokens=default_max_new_tokens,
            enable_grammar=enable_grammar,
        )
        self.cfg = cfg
        self.model_path = Path(cfg.model_path)
        self.device = device or "cpu"

        self.llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.cfg.n_gpu_layers,
            n_ctx=self.cfg.n_ctx,
            n_batch=self.cfg.n_batch,
            n_threads=self.cfg.n_threads,
            flash_attn=self.cfg.flash_attn,
            verbose=self.cfg.verbose,
            seed=self.cfg.seed,
        )

        if self.cfg.enable_grammar and LlamaGrammar is not None:
            self.json_grammar = LlamaGrammar.from_string(JSON_SCHEMA_GBNF)
        else:
            self.json_grammar = None

    @staticmethod
    def _extract_json_payload(text: str) -> object | None:
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
            args_json = args if isinstance(args, str) else json.dumps(args)
            normalized.append(
                {
                    "id": str(call.get("id", f"call_{idx + 1}")),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_json,
                    },
                }
            )

        return normalized or None

    @staticmethod
    def _prepare_messages_for_llama(messages: list[dict[str, object]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", "") or "")

            if role == "tool":
                tool_name = str(msg.get("name", "tool"))
                normalized.append({"role": "user", "content": f"[Tool result: {tool_name}] {content}"})
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
        temperature = self.cfg.default_temperature if temperature is None else temperature
        max_new_tokens = self.cfg.default_max_new_tokens if max_new_tokens is None else max_new_tokens

        if messages is None:
            if prompt is None:
                raise ValueError("Either messages or prompt must be provided.")
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages = self._prepare_messages_for_llama(messages)

        if stage == "glossary":
            # Stabilize structure for tool-calling/json extraction.
            temperature = 0.0

        create_kwargs: dict[str, object] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if json_mode and self.json_grammar is not None:
            create_kwargs["grammar"] = self.json_grammar

        response = self.llm.create_chat_completion(**create_kwargs)
        choice = response["choices"][0]["message"]
        text = str(choice.get("content", "") or "").strip()

        tool_calls = None
        raw_tool_calls = choice.get("tool_calls")
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            tool_calls = raw_tool_calls
        elif stage == "glossary":
            parsed = self._extract_json_payload(text)
            tool_calls = self._normalize_tool_calls(parsed)
            if tool_calls:
                text = ""

        usage_raw = response.get("usage", {}) if isinstance(response, dict) else {}
        usage = LLMUsage(
            prompt_tokens=usage_raw.get("prompt_tokens"),
            completion_tokens=usage_raw.get("completion_tokens"),
            total_tokens=usage_raw.get("total_tokens"),
        )

        return LLMResult(
            text=text,
            model=str(self.model_path),
            usage=usage,
            tool_calls=tool_calls,
            raw=response if isinstance(response, dict) else {"response": str(response)},
        )
