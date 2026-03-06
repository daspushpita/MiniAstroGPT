from dataclasses import dataclass
import json
import os
import re
import time
from datetime import datetime
from typing import Any
from .prompts import Prompts
from miniastrolm.llm.base import LLMClient
from miniastrolm.agent.validators import Validator

@dataclass
class AgentRun:
    mode: str
    plan: str
    draft: str
    glossary: str
    critic: str
    revised_draft: str
    
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

class AstroAgent:

    def __init__(self, llm_client: LLMClient, max_turns: int, 
                max_revision_attempts: int,
                threshold_hallucination: int, 
                threshold_clarity: int,
                threshold_structure: int,
                fast_mode: bool | None = None,
                fast_max_new_tokens: int = 700) -> None:
        
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.TOOL_MAPPING = {
            "google_search": self._google_search_tool,
        }
        self.threshold_hallucination = threshold_hallucination
        self.threshold_clarity = threshold_clarity
        self.threshold_structure = threshold_structure
        self.max_revision_attempts = max_revision_attempts
        env_fast_mode = os.getenv("ASTRO_FAST_MODE")
        if fast_mode is None:
            fast_mode = str(env_fast_mode or "").strip().lower() in {"1", "true", "yes", "on"}
        self.fast_mode = bool(fast_mode)
        self.fast_max_new_tokens = fast_max_new_tokens

    @staticmethod
    def _google_search_tool(query: str) -> str:
        from miniastrolm.tools.google_search import google_search
        return google_search(query)

    def _execute_tool_call(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call based on the tool name and arguments.
        """
        if tool_name not in self.TOOL_MAPPING:
            return "Tool error: unsupported tool."
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "Tool error: missing 'query' argument."

        tool_func = self.TOOL_MAPPING.get(tool_name)
        if tool_func is None:
            return "Tool error: unsupported tool."

        try:
            return tool_func(query)
        except Exception as e:
            return f"Tool error: {e}"

    # helper: parse score robustly
    def _as_int(self, x: Any, key: str) -> tuple[int | None, str | None]:
        if isinstance(x, int):
            return x, None
        if isinstance(x, float) and x.is_integer():
            return int(x), None
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip()), None
        return None, f"Critic key '{key}' must be an integer 0-5."

    def _validate_critic_payload(self, payload: Any) -> tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "Critic output must be a JSON object."

        required_keys = {"scores", "hallucinated_claims", "fix_instructions"}

        scores = payload.get("scores")
        if not isinstance(scores, dict):
            return False, "Critic key 'scores' must be a JSON object."
        
        score_keys = {"hallucination", "structure", "clarity"}
        missing_scores = score_keys - set(scores.keys())
        if missing_scores:
            return False, f"Critic JSON missing score keys: {sorted(missing_scores)}"
        
        hall, err = self._as_int(scores.get("hallucination"), "scores.hallucination")
        if err: return False, err
        struct, err = self._as_int(scores.get("structure"), "scores.structure")
        if err: return False, err
        clar, err = self._as_int(scores.get("clarity"), "scores.clarity")
        if err: return False, err

        for key, val in [("scores.hallucination", hall), ("scores.structure", struct), ("scores.clarity", clar)]:
            if not (0 <= val <= 5):
                return False, f"Critic key '{key}' must be in range 0-5."

        hallucinated_claims = payload.get("hallucinated_claims")
        if not isinstance(hallucinated_claims, list) or any(not isinstance(x, str) for x in hallucinated_claims):
            return False, "Critic key 'hallucinated_claims' must be a list of strings."

        fix_instructions = payload.get("fix_instructions")
        if not isinstance(fix_instructions, list) or any(not isinstance(x, str) for x in fix_instructions):
            return False, "Critic key 'fix_instructions' must be a list of strings."

        if (hall >= 3 or struct >= 3 or clar >= 3) and len(fix_instructions) == 0:
            return False, "Critic must provide fix_instructions when any score >= 3."

        return True, ""

    def _extract_first_json_object(self, text: str) -> dict[str, Any] | None:
        """Try hard to extract the first JSON object from an LLM response."""
        raw = (text or "").strip()
        if not raw:
            return None

        candidates = [raw]
        # pull content out of ```json ... ``` blocks if present
        candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw))

        decoder = json.JSONDecoder()
        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue

            # 1) direct parse
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

            # 2) raw_decode from first '{'
            for i, ch in enumerate(cand):
                if ch != "{":
                    continue
                try:
                    obj, _ = decoder.raw_decode(cand[i:])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue

        return None


    def _run_critic(self, abstract: str, draft: str, max_attempts: int = 3) -> tuple[dict[str, Any] | None, str]:
        critic_prompt = Prompts(mode="critic").build_critic_prompt(abstract, draft)
        system_critic_prompt = (
            "You are a strict editorial reviewer evaluating a magazine-style explanation of an astronomy abstract. "
            "Return ONLY valid JSON matching the schema."
        )

        last_error = "Unknown critic failure."
        for _ in range(max(1, max_attempts)):
            critic_result = self.llm_client.generate(
                prompt=critic_prompt,
                stage="critic",
                system_prompt=system_critic_prompt,
                temperature=0.0,
                max_new_tokens=512,
                json_mode=True,   # keep this ON if your client supports it
            )

            critic_text = (critic_result.text or "").strip()
            payload = self._extract_first_json_object(critic_text)
            if payload is None:
                last_error = "Critic returned invalid JSON."
                continue

            is_valid, error_msg = self._validate_critic_payload(payload)
            if is_valid:
                return payload, ""

            last_error = error_msg

        return None, last_error

    @staticmethod
    def _normalize_glossary_output(text: str) -> str:
        """Extract the first JSON object from model output and return normalized JSON."""
        raw = (text or "").strip()
        if not raw:
            return "{}"

        candidates = [raw]
        candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw))
        decoder = json.JSONDecoder()

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue

            parsed_obj: Any | None = None
            try:
                parsed_obj = json.loads(candidate)
            except json.JSONDecodeError:
                for idx, ch in enumerate(candidate):
                    if ch != "{":
                        continue
                    try:
                        parsed_obj, _ = decoder.raw_decode(candidate[idx:])
                        break
                    except json.JSONDecodeError:
                        continue

            if isinstance(parsed_obj, dict):
                normalized: dict[str, str] = {}
                for k, v in parsed_obj.items():
                    if isinstance(k, str) and isinstance(v, str):
                        normalized[k] = v.strip()
                return json.dumps(normalized, ensure_ascii=False)

        return "{}"
        
    def glossary_agent(self, abstract: str, max_tool_turns: int) -> str:
        # Generate the glossary
        glossary_prompt = Prompts(mode="glossary").build_glossary_prompt(abstract)
        glossary_system_prompt = f"""You are an expert astronomer and science communicator. Your task is to identify any technical terms in the given abstract that might be difficult for a general audience to understand, and provide simple definitions for those terms."""

        messages = [{"role": "system", "content": glossary_system_prompt},
                    {"role": "user", "content": glossary_prompt}]
        
        for _ in range(max_tool_turns):
            glossary_result = self.llm_client.generate(
                messages=messages,
                stage = "glossary",
                temperature = 0.2,
                max_new_tokens = 512,
            )
            tool_calls = glossary_result.tool_calls or []

            if not tool_calls:
                return self._normalize_glossary_output(glossary_result.text.strip())
            
            messages.append(
                {"role": "assistant",
                "content": glossary_result.text or "",
                "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                function_block = call.get("function", {})
                tool_name = function_block.get("name", "")
                arguments_json = function_block.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_json or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                result = self._execute_tool_call(tool_name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": tool_name,
                        "content": result,
                    }
                )

        return "{}"
    
    def _critic_passes(self, critic_payload: dict[str, Any]) -> bool:
        scores = critic_payload.get("scores", {})
        hallucination = int(scores.get("hallucination", 5))
        structure = int(scores.get("structure", 5))
        clarity = int(scores.get("clarity", 5))
        return (hallucination <= 1) and (structure <= 2) and (clarity <= 2)
    
    def _validator_to_critic(self, v) -> dict[str, Any]:
    # v is your ValidatorResult
        return {
            "scores": {"hallucination": 0, "structure": 3, "clarity": 1},
            "fix_instructions": [
                f"Fix formatting/constraint failures: {', '.join(v.failures)}.",
                f"Keep exactly 4 paragraphs and 180–250 words (current: {v.word_count} words, {v.paragraph_count} paragraphs).",
            ],
            "hallucinated_claims": [],
        }
    
    def run(self, abstract: str) -> AgentRun:
        """Run the whole series of agents with the given abstract.

        Args:
            abstract (str): The abstract to process.

        Returns:
            AgentRun: The result of running the agent.
        """
        def _log_timing(stage: str, event: str, elapsed: float | None = None) -> None:
            ts = datetime.now().isoformat(timespec="seconds")
            if elapsed is None:
                print(f"[TIMING] {ts} | {stage} | {event}")
            else:
                print(f"[TIMING] {ts} | {stage} | {event} | {elapsed:.2f}s")

        _log_timing("run", "start")

        # ----------------------------
        # FAST MODE: single prompt + single LLM call
        # ----------------------------
        if self.fast_mode:
            _log_timing("fast_mode", "enabled")
            stage_t0 = time.perf_counter()
            _log_timing("prompt", "start")
            fast_prompt = Prompts(mode="write").build_fast_explanation_prompt(abstract)
            _log_timing("prompt", "end", time.perf_counter() - stage_t0)

            stage_t0 = time.perf_counter()
            _log_timing("revised_draft", "start")
            system_writer_prompt = (
                "You are an expert astronomer and science communicator. "
                "Write a clear, concise magazine-style explanation of the abstract."
            )
            fast_result = self.llm_client.generate(
                prompt=fast_prompt,
                stage="write",
                system_prompt=system_writer_prompt,
                temperature=0.2,
                max_new_tokens=self.fast_max_new_tokens,
                json_mode=False,
            )
            explanation = (fast_result.text or "").strip()
            _log_timing("revised_draft", "end", time.perf_counter() - stage_t0)
            _log_timing("run", "end")
            return AgentRun(
                mode="fast",
                plan="",
                draft=explanation,
                glossary="{}",
                critic="{}",
                revised_draft=explanation,
            )

        # ----------------------------
        # 1) Plan
        # ----------------------------
        stage_t0 = time.perf_counter()
        _log_timing("plan", "start")
        plan_prompt = Prompts(mode="plan").build_planner_prompt(abstract)
        system_plan_prompt = (
            "You are an expert astronomer and science communicator. "
            "Create a clear, concise plan for rewriting the abstract into a magazine-style explanation."
        )
        plan_result = self.llm_client.generate(
            prompt=plan_prompt,
            stage="plan",
            system_prompt=system_plan_prompt,
            temperature=0.2,
            max_new_tokens=512,
        )
        plan = (plan_result.text or "").strip()
        _log_timing("plan", "end", time.perf_counter() - stage_t0)

        # ----------------------------
        # 2) Glossary (tool-using)
        # ----------------------------
        stage_t0 = time.perf_counter()
        _log_timing("glossary", "start")
        glossary = self.glossary_agent(abstract=abstract, max_tool_turns=self.max_turns)
        _log_timing("glossary", "end", time.perf_counter() - stage_t0)

        # ----------------------------
        # 3) Write first draft
        # ----------------------------
        stage_t0 = time.perf_counter()
        _log_timing("draft", "start")
        writer_prompt = Prompts(mode="write").build_writer_prompt(abstract, plan, glossary)
        system_writer_prompt = (
            "You are an expert astronomer and science communicator. "
            "Write a clear, concise magazine-style explanation of the abstract, following the plan."
        )
        writer_result = self.llm_client.generate(
            prompt=writer_prompt,
            stage="write",
            system_prompt=system_writer_prompt,
            temperature=0.2,
            max_new_tokens=800,
            json_mode=False,
        )
        draft = (writer_result.text or "").strip()
        _log_timing("draft", "end", time.perf_counter() - stage_t0)

        # ----------------------------
        # 4) Repair-to-spec loop (fix validator failures BEFORE critic)
        # ----------------------------
        current_draft = draft
        v = Validator(draft=current_draft, min_words=180, max_words=250, required_paragraphs=4).validate()

        repair_attempts = 2
        last_validator = v

        for attempt in range(repair_attempts):
            if v.passed:
                break

            # Build synthetic critic payload from validator failures, in the *new* critic schema
            synthetic_critique = self._validator_to_critic(v)
            synthetic_critique_json = json.dumps(synthetic_critique, ensure_ascii=False)

            repair_prompt = Prompts(mode="revise").build_reviser_prompt(
                abstract=abstract,
                plan=plan,
                draft=current_draft,
                critique_json=synthetic_critique_json,
                glossary=glossary,
            )
            repair_system_prompt = (
                "You are an expert astronomer and science communicator. "
                "Revise the draft to satisfy the constraints in fix_instructions. "
                "Do not add new facts."
            )
            stage_t0 = time.perf_counter()
            _log_timing("revised_draft", f"repair_attempt_{attempt + 1}_start")
            repair_result = self.llm_client.generate(
                prompt=repair_prompt,
                stage="revise",
                system_prompt=repair_system_prompt,
                temperature=0.2,
                max_new_tokens=800,
                json_mode=False,
            )
            _log_timing("revised_draft", f"repair_attempt_{attempt + 1}_end", time.perf_counter() - stage_t0)
            current_draft = (repair_result.text or "").strip()
            v = Validator(draft=current_draft, min_words=180, max_words=250, required_paragraphs=4).validate()
            last_validator = v

        if not v.passed:
            _log_timing("run", "end")
            return AgentRun(
                mode="validation_failed",
                plan=plan,
                draft=draft,
                glossary=glossary,  # keep for debugging
                critic=json.dumps(
                    {
                        "validator_failures": last_validator.failures,
                        "word_count": last_validator.word_count,
                        "paragraph_count": last_validator.paragraph_count,
                        "forbidden_hits": last_validator.forbidden_hits,
                    },
                    ensure_ascii=False,
                ),
                revised_draft=current_draft,
            )

        # Use repaired-to-spec draft as the base draft
        draft = current_draft
        # ----------------------------
        # 5) Critic (scores)
        # ----------------------------
        stage_t0 = time.perf_counter()
        _log_timing("critic", "start")
        critic_payload, critic_error = self._run_critic(abstract=abstract, draft=draft, max_attempts=2)
        _log_timing("critic", "end", time.perf_counter() - stage_t0)
        if critic_payload is None:
            critic_error_payload = {
                "scores": {"hallucination": 5, "structure": 5, "clarity": 5},
                "fix_instructions": [f"Critic failed schema validation: {critic_error}"],
                "hallucinated_claims": [],
            }
            _log_timing("run", "end")
            return AgentRun(
                mode="critic_invalid",
                plan=plan,
                draft=draft,
                glossary=glossary,
                critic=json.dumps(critic_error_payload, ensure_ascii=False),
                revised_draft=draft,
            )

        critic_json = json.dumps(critic_payload, ensure_ascii=False)
        if self._critic_passes(critic_payload):
            _log_timing("run", "end")
            return AgentRun(
                mode="final",
                plan=plan,
                draft=draft,
                glossary=glossary,
                critic=critic_json,
                revised_draft=draft,
            )

        # ----------------------------
        # 6) Revise loop (validator + critic until pass or attempts exhausted)
        # ----------------------------
        current_draft = draft
        current_critic_payload = critic_payload
        current_critic_json = critic_json
        for attempt in range(self.max_revision_attempts):
            revision_prompt = Prompts(mode="revise").build_reviser_prompt(
                abstract=abstract,
                plan=plan,
                draft=current_draft,
                critique_json=current_critic_json,
                glossary=glossary,
            )
            system_revision_prompt = (
                "You are an expert astronomer and science communicator. "
                "Revise the draft using fix_instructions as a checklist. "
                "Remove every item listed in hallucinated_claims. "
                "Do not add new facts beyond the abstract."
            )
            stage_t0 = time.perf_counter()
            _log_timing("revised_draft", f"attempt_{attempt + 1}_start")
            revision_result = self.llm_client.generate(
                prompt=revision_prompt,
                stage="revise",
                system_prompt=system_revision_prompt,
                temperature=0.2,
                max_new_tokens=800,
                json_mode=False,
            )
            _log_timing("revised_draft", f"attempt_{attempt + 1}_end", time.perf_counter() - stage_t0)
            revised_draft = (revision_result.text or "").strip()

            revised_validate = Validator(
                draft=revised_draft,
                min_words=180,
                max_words=250,
                required_paragraphs=4,
            ).validate()

            if not revised_validate.passed:
                # Convert validator failures into the same critic schema and keep iterating
                current_draft = revised_draft
                current_critic_payload = self._validator_to_critic(revised_validate)
                current_critic_json = json.dumps(current_critic_payload, ensure_ascii=False)
                continue

            stage_t0 = time.perf_counter()
            _log_timing("critic", f"revised_attempt_{attempt + 1}_start")
            revised_critic_payload, revised_critic_error = self._run_critic(
                abstract=abstract,
                draft=revised_draft,
                max_attempts=2,
            )
            _log_timing("critic", f"revised_attempt_{attempt + 1}_end", time.perf_counter() - stage_t0)
            if revised_critic_payload is None:
                critic_error_payload = {
                    "scores": {"hallucination": 5, "structure": 5, "clarity": 5},
                    "fix_instructions": [f"Critic failed schema validation: {revised_critic_error}"],
                    "hallucinated_claims": [],
                }
                _log_timing("run", "end")
                return AgentRun(
                    mode="critic_invalid",
                    plan=plan,
                    draft=draft,
                    glossary=glossary,
                    critic=json.dumps(critic_error_payload, ensure_ascii=False),
                    revised_draft=revised_draft,
                )

            current_draft = revised_draft
            current_critic_payload = revised_critic_payload
            current_critic_json = json.dumps(revised_critic_payload, ensure_ascii=False)

            if self._critic_passes(revised_critic_payload):
                _log_timing("run", "end")
                return AgentRun(
                    mode="final",
                    plan=plan,
                    draft=draft,
                    glossary=glossary,
                    critic=current_critic_json,
                    revised_draft=revised_draft,
                )

        _log_timing("run", "end")
        return AgentRun(
            mode="needs_revision",
            plan=plan,
            draft=draft,
            glossary=glossary,
            critic=json.dumps(current_critic_payload, ensure_ascii=False),
            revised_draft=current_draft,
        )
