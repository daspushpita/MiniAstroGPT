from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, sqlite3
from collections import Counter
from typing import Dict, Iterator, List, Tuple, Any, Optional
from pathlib import Path
from miniastrolm.llm.teacher import Validation_Regeneration



@dataclass
class JudgeResult:
    """Structured result from judge."""
    is_valid: bool
    errors: List[str]
    obj: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]
    
@dataclass
class JudgeConfig:
    """Hyperparameters + thresholds for judge behavior."""
    max_attempts: int = 3
    max_new_tokens: int = 512
    do_sample: bool = False
    
    min_faithfulness_to_keep: int = 4
    min_overall_to_keep: int = 60
    
    REQUIRED_TOP_KEYS = {"id", "scores", "error_tags", "rationale", "rewrite_hint"}
    REQUIRED_SCORE_KEYS = {"faithfulness", "clarity", "jargon", "structure", "uncertainty", "overall"}
    CANONICAL_TAGS = {"hallucination", "insufficient_coverage","style_drift", "overlap", "truncation", "non_json", "generic_filler"}

class JudgePromptBuilder:
    """Builds the evaluation prompt given abstract + explanation."""
    
    def build(self, *, paper_id: str, abstract: str, explanation: str) -> str:
        return f"""
You are a strict evaluator for scientific-to-layman explanations.

SOURCE OF TRUTH (ABSTRACT):
{abstract}

TEACHER EXPLANATION TO EVALUATE:
{explanation}

TASK:
Score the explanation using these integer scores:
- faithfulness (0-5): matches the abstract, no unsupported claims
- clarity (0-5): easy to understand
- jargon (0-5): amount of technical jargon (higher = more jargon)
- structure (0-5): logical flow and organization
- uncertainty (0-5): appropriate uncertainty/hedging
- overall (0-100): overall quality

HALLUCINATION RULE:
If the explanation contains claims NOT supported by the abstract:
- include "hallucination" in error_tags
- set faithfulness <= 2

OUTPUT RULES (HARD):
- Output EXACTLY ONE JSON object and NOTHING ELSE.
- Do not include steps, analysis, or headings. Output JSON only.
- Your FIRST character must be '{{' and your LAST character must be '}}'.
- Do NOT use markdown or triple backticks.
- No extra keys besides: id, scores, error_tags, rationale, rewrite_hint
- All scores must be integers (overall 0-100; others 0-5).
- error_tags must be a JSON list of strings.
- rationale must be exactly 1 sentence.
- rewrite_hint must be exactly 1 concrete actionable sentence.
- id must equal: "{paper_id}"

RETURN THIS JSON (exact keys):
{{
    "id": "{paper_id}",
    "scores": {{
    "faithfulness": 0,
    "clarity": 0,
    "jargon": 0,
    "structure": 0,
    "uncertainty": 0,
    "overall": 0
    }},
    "error_tags": [],
    "rationale": "",
    "rewrite_hint": ""
}}
""".strip()


class JudgePromptBuilder_v2:
    """Builds the evaluation prompt given abstract + explanation."""
    
    def build(self, *, paper_id: str, abstract: str, explanation: str) -> str:
        return f"""
You are a strict evaluator for scientific-to-layman explanations.

SOURCE OF TRUTH (ABSTRACT):
{abstract}

TEACHER EXPLANATION TO EVALUATE:
{explanation}

TASK:
Score the explanation using these integer scores:
- faithfulness (0-5): matches the abstract, no unsupported claims
- clarity (0-5): easy to understand
- jargon (0-5): use of unexplained technical jargon or acronyms (higher = more unexplained jargon; explained terms should not be penalized)
- structure (0-5): logical flow and organization
- uncertainty (0-5): appropriate uncertainty/hedging
- overall (0-100): overall quality

DEFINITIONS (CRITICAL):

A "hallucination" means introducing new factual claims, results, mechanisms,
named entities, numerical values, or conclusions that are NOT supported by the abstract.

Do NOT tag hallucination for:
- general background explanations of concepts that appear in the abstract
- defining technical terms mentioned in the abstract
- high-level conceptual framing intended for a non-expert reader
- missing depth or missing topics the abstract does not emphasize

If hallucination is present:
- include "hallucination" in error_tags
- set faithfulness <= 2

If content is missing or underexplained:
- use error tag "insufficient_coverage"
- do NOT reduce faithfulness below 3 for this reason alone

TEXT FIELDS (VERY STRICT):
- rationale must be EXACTLY 1 short sentence <= 20 words.
- rewrite_hint must be EXACTLY 1 short actionable sentence <= 25 words.
- Do NOT add examples, extra clauses, greetings, sign-offs, or any extra text.
- Do NOT include markdown code fences.

FAITHFULNESS SCORING RULE:
Faithfulness measures consistency with the abstract.
Faithfulness must NOT be set to 0 unless the explanation is largely unrelated
to the abstract or contains multiple fabricated claims.

SCORING RULE:
Each score must be assigned independently.
A weakness in one dimension must NOT automatically zero out other scores or the overall score.

OVERALL SCORE GUIDANCE:
Overall reflects usefulness and quality for a non-expert reader.
Explanatory background is allowed if it does not introduce unsupported claims.

OUTPUT RULES (HARD):
- Output EXACTLY ONE JSON object and NOTHING ELSE.
- Do not include steps, analysis, or headings. Output JSON only.
- Your FIRST character must be '{{' and your LAST character must be '}}'.
- Do NOT use markdown or triple backticks.
- No extra keys besides: id, scores, error_tags, rationale, rewrite_hint
- All scores must be integers (overall 0-100; others 0-5).
- error_tags must be a JSON list of strings.
- rationale must be exactly 1 sentence.
- rewrite_hint must be exactly 1 concrete actionable sentence.
- id must equal: "{paper_id}"

IMPORTANT:
Return ONLY the JSON object. No preface like "I'll correct..." and no trailing text.

RETURN THIS JSON (exact keys):
{{
    "id": "{paper_id}",
    "scores": {{
    "faithfulness": 0,
    "clarity": 0,
    "jargon": 0,
    "structure": 0,
    "uncertainty": 0,
    "overall": 0
    }},
    "error_tags": [],
    "rationale": "",
    "rewrite_hint": ""
}}
""".strip()

class JudgeValidator:
    """Validates and parses judge JSON output."""
    def __init__(self, model_name):
        
        self.model_name = model_name
        self.my_config = JudgeConfig()

    def validate(self, raw_text: str, paper_id: str):
        errors, metrics = [], {}
        # text = (raw_text or "").strip()

        obj = JudgeJsonExtractor(raw_text, paper_id).extract_obj()
        if obj is None:
            return JudgeResult(False, ["Invalid or missing JSON"], None, metrics)
        
        # json_text = self.my_validator._attempt_autofix_to_json(text, paper_id, mode="judge")        
        # if not json_text:
        #     return JudgeResult(is_valid=False, errors=["Empty output from teacher"], obj=None, metrics=metrics)
        
        # try:
        #     obj = json.loads(json_text)
        # except json.JSONDecodeError:
        #     return JudgeResult(False, ["Invalid JSON"], None, metrics)

        valid, msg = self._check_required_keys(obj)
        if not valid:
            errors.append(msg)
        valid, msg = self._check_id(obj.get("id", ""), paper_id=paper_id)
        if not valid:
            errors.append(msg)
        valid, msg = self._check_scores(obj.get("scores", {}))
        scores = obj.get("scores", {})
        if isinstance(scores, dict):
            metrics["faithfulness"] = scores.get("faithfulness")
            metrics["overall"] = scores.get("overall")
        else:
            metrics["faithfulness"] = None
            metrics["overall"] = None
        if not valid:
            errors.append(msg)
        valid, msg = self._check_error_tags(obj.get("error_tags", []))
        error_tags = obj.get("error_tags", [])
        metrics["num_error_tags"] = len(error_tags) if isinstance(error_tags, list) else None

        if not valid:
            errors.append(msg)
        valid, msg = self._check_rationale(obj.get("rationale", ""))
        if not valid:
            errors.append(msg)
        valid, msg = self._check_rewrite_hint(obj.get("rewrite_hint", ""))
        if not valid:
            errors.append(msg)
            
        degenerate_errors = self._check_degenerate(obj)
        errors.extend(degenerate_errors)
        
        return JudgeResult(is_valid=(len(errors) == 0), errors=errors, obj=obj, metrics=metrics)

    def _check_required_keys(self, obj: Dict[str, Any]) -> tuple[bool, str]:
        req_keys = JudgeConfig.REQUIRED_TOP_KEYS
        missing =[k for k in req_keys if k not in obj]
        if missing:
            return (False, f"Missing required keys: {missing}")
        return (True, "")

    def _check_id(self, extracted_id: Any, *, paper_id: str) -> tuple[bool, str]:
        if extracted_id != paper_id:
            return (False, f"ID mismatch: expected {paper_id}, got {extracted_id}")
        return (True, "")

    def _check_scores(self, scores: Dict[str, Any]) -> tuple[bool, str]:
        score_keys = JudgeConfig.REQUIRED_SCORE_KEYS
        for key in score_keys:
            if key not in scores:
                return (False, f"Missing score for: {key}")
            value = scores[key]
            if not isinstance(value, int):
                return (False, f"Score for {key} is not an interger")
            if key == "overall":
                if not (0 <= value <= 100):
                    return (False, f"Overall score {value} out of range (0-100)")
            else:
                if not (0 <= value <= 5):
                    return (False, f"Score for {key} {value} out of range (0-5)")

        return (True, "")

    def _check_error_tags(self, error_tags: Any) -> tuple[bool, str]:
        if not isinstance(error_tags, list):
            return (False, "error_tags is not a list")
        for tag in error_tags:
            if not isinstance(tag, str):
                return (False, f"error_tag {tag} is not a string")
        bad = [t for t in error_tags if t.split(":", 1)[0].strip().lower() not in JudgeConfig.CANONICAL_TAGS]
        if bad:
            return (False, f"Non-canonical error_tags: {bad}")
        return (True, "")
    
    # def _check_rationale(self, rationale: Any) -> List[Any]:
    #     if not isinstance(rationale, str):
    #         return [False, "rationale is not a string"]
    #     text = rationale.strip()
    #     if ((len(text.split(".")) > 2) or (len(text.split("!")) > 2) or (len(text.split("?")) > 2)):
    #         return [False, "Rationale is more than 1 sentence"]
    #     return [True, ""]
    
    # def _check_rewrite_hint(self, rewrite_hint: Any) -> List[Any]:
    #     if not isinstance(rewrite_hint, str):
    #         return [False, "rewrite_hint is not a string"]
    #     text = rewrite_hint.strip()
    #     if ((len(text.split(".")) > 2) or (len(text.split("!")) > 2) or (len(text.split("?")) > 2)):
    #         return [False, "rewrite_hint is more than 1 sentence"]
    #     return [True, ""]
    
    def _check_one_sentenceish(self, s: str, *, max_words: int) -> tuple[bool, str]:
        s = s.strip()
        if not s:
            return (False, "Empty text")
        if len(s.split()) > max_words:
            return (False, f"Too long (> {max_words} words)")
        # optionally: reject newline, reject multiple sentences via regex on [.!?] count
        punct = sum(s.count(ch) for ch in ".!?")
        if punct > 2:
            return (False, "Too many sentence terminators")
        return (True, "")
    
    def _check_rationale(self, rationale: Any) -> tuple[bool, str]:
        if not isinstance(rationale, str):
            return (False, "rationale is not a string")
        return self._check_one_sentenceish(rationale, max_words=20)

    def _check_rewrite_hint(self, rewrite_hint: Any) -> tuple[bool, str]:
        if not isinstance(rewrite_hint, str):
            return (False, "rewrite_hint is not a string")
        return self._check_one_sentenceish(rewrite_hint, max_words=25)

    def _check_degenerate(self, obj: Dict[str, Any]) -> List[str]:
        errs = []
        scores = obj.get("scores", {})
        if not isinstance(scores, dict):
            return ["DEGENERATE:scores_not_dict"]

        # all-zero score collapse
        keys = ["faithfulness", "clarity", "jargon", "structure", "uncertainty", "overall"]
        vals = [scores.get(k) for k in keys]
        if all(isinstance(v, int) and v == 0 for v in vals):
            errs.append("DEGENERATE:all_scores_zero")

        # empty rationale/hint (even if schema-valid)
        rat = obj.get("rationale", "")
        hint = obj.get("rewrite_hint", "")
        if isinstance(rat, str) and not rat.strip():
            errs.append("DEGENERATE:empty_rationale")
        if isinstance(hint, str) and not hint.strip():
            errs.append("DEGENERATE:empty_rewrite_hint")

        # hallucination consistency constraint
        tags = obj.get("error_tags", [])
        if isinstance(tags, list) and "hallucination" in tags:
            f = scores.get("faithfulness")
            if isinstance(f, int) and f > 2:
                errs.append("CONSISTENCY:hallucination_requires_faithfulness_le2")

        return errs
    
class JudgeRepairPromptBuilder:
    def build(self,*,paper_id: str,original_prompt: str,
        errors: List[str],
        bad_output: str) -> str:
        
        err_block = "\n".join(f"- {e}" for e in errors[:8])
        return f"""{original_prompt}
Your previous output was invalid.

Fix these issues:
{err_block}

Rules:
- Output ONLY a single JSON object (no markdown, no commentary).
- Do not include steps, analysis, or headings. Output JSON only.
- Do NOT wrap in markdown or triple backticks.
- Use EXACT keys: id, scores, error_tags, rationale, rewrite_hint
- All scores must be integers.
- rationale = 1 sentence; rewrite_hint = 1 actionable sentence.
- id must equal: {paper_id}

Bad output:
{bad_output}
""".strip()

class JudgeRepairPromptBuilder_v2:
    def build(self, *, paper_id: str, original_prompt: str, errors: List[str], bad_output: str) -> str:
        err_block = "\n".join(f"- {e}" for e in errors[:8])

        return f"""{original_prompt}

Your previous output was invalid JSON or violated the required schema.

You are repairing formatting ONLY.
Do NOT re-evaluate the explanation. Preserve the original scores/tags/rationale as much as possible
and only change what is required to satisfy the rules below.

Fix these issues:
{err_block}

HARD RULES:
- Output ONLY a single JSON object (no markdown, no commentary).
- Use EXACT top-level keys: id, scores, error_tags, rationale, rewrite_hint
- Remove any extra keys.
- id must equal: "{paper_id}"

SCORES:
- scores must be an object with EXACT keys:
  faithfulness, clarity, jargon, structure, uncertainty, overall
- faithfulness/clarity/jargon/structure/uncertainty must be integers 0-5
- overall must be an integer 0-100

ERROR TAGS:
- error_tags must be a JSON list of strings
- Only use these canonical tags (drop any others):
  ["hallucination","insufficient_coverage","style_drift","generic_filler","overlap","truncation","non_json"]

CONSISTENCY CONSTRAINT:
- If "hallucination" is present in error_tags, set scores.faithfulness = min(scores.faithfulness, 2)

TEXT FIELDS:
- rationale must be EXACTLY 1 sentence.
- rewrite_hint must be EXACTLY 1 concrete actionable sentence.

Bad output:
{bad_output}

Return ONLY the repaired JSON object.
""".strip()

class JudgeReevalPromptBuilder:
    """Used when the judge output is missing/degenerate and needs re-evaluation."""
    def build(self, *, paper_id: str, abstract: str, explanation: str, reason: str) -> str:
        return f"""
You previously failed to produce a usable evaluation (reason: {reason}).
Re-evaluate from scratch.

SOURCE OF TRUTH (ABSTRACT):
{abstract}

TEACHER EXPLANATION TO EVALUATE:
{explanation}

TASK:
Score the explanation using these integer scores:
- faithfulness (0-5): consistency with the abstract; no unsupported claims
- clarity (0-5): easy to understand
- jargon (0-5): unexplained technical jargon or acronyms (higher = more unexplained jargon)
- structure (0-5): logical flow and organization
- uncertainty (0-5): appropriate uncertainty/hedging
- overall (0-100): overall quality for a non-expert reader

DEFINITIONS (CRITICAL):
- "hallucination" means introducing new factual claims, results, mechanisms, named entities,
  numerical values, or conclusions NOT supported by the abstract.
- Do NOT tag hallucination for: background definitions of terms in the abstract, conceptual framing,
  or missing coverage.
- If hallucination is present: include "hallucination" in error_tags AND set faithfulness <= 2.
- If the explanation is missing important details or is underexplained: use "insufficient_coverage"
  (this is NOT hallucination).

SCORING RULES:
- Scores must be assigned independently (do not zero everything due to one issue).
- Faithfulness must NOT be 0 unless the explanation is largely unrelated to the abstract
  or contains multiple fabricated claims.

OUTPUT RULES (HARD):
- Output EXACTLY ONE JSON object and NOTHING ELSE.
- Your FIRST character must be '{{' and your LAST character must be '}}'.
- No markdown, no backticks, no extra text.
- No extra keys besides: id, scores, error_tags, rationale, rewrite_hint
- All scores must be integers (overall 0-100; others 0-5).
- error_tags must be a JSON list of strings.
- rationale must be exactly 1 sentence.
- rewrite_hint must be exactly 1 concrete actionable sentence.
- id must equal: "{paper_id}"

RETURN THIS JSON (exact keys):
{{
  "id": "{paper_id}",
  "scores": {{
    "faithfulness": 0,
    "clarity": 0,
    "jargon": 0,
    "structure": 0,
    "uncertainty": 0,
    "overall": 0
  }},
  "error_tags": [],
  "rationale": "",
  "rewrite_hint": ""
}}
""".strip()

class LLMJudge:
    """
    Orchestrates:
      prompt → llm → validate → retry → final score object
    """

    def __init__(self, llm_client: Any, config: JudgeConfig | None = None, 
                prompt_builder: JudgePromptBuilder | None = None,
                validator: JudgeValidator | None = None,
                repair_prompt_builder: JudgeRepairPromptBuilder | None = None,
                reeval_prompt_builder: JudgeReevalPromptBuilder | None = None,
                max_attempts: int = 3):
        
        self.llm_client = llm_client
        self.config = config or JudgeConfig()
        self.prompt_builder = prompt_builder or JudgePromptBuilder_v2()
        self.validator = validator or JudgeValidator(model_name=self.llm_client.model_id)
        self.repair_prompt_builder = repair_prompt_builder or JudgeRepairPromptBuilder_v2()
        self.reeval_prompt_builder = reeval_prompt_builder or JudgeReevalPromptBuilder()
        self.max_attempts = max_attempts

    def judge_one(self, *, paper_id: str, abstract: str, explanation: str) -> Dict[str, Any]:
        
        if isinstance(explanation, dict):
            explanation = explanation.get("explanation", "")
        elif not isinstance(explanation, str):
            explanation = str(explanation)
        original_prompt = self.prompt_builder.build(paper_id=paper_id, abstract=abstract, explanation=explanation[:2000])
        prompt = original_prompt
        
        attempt = 0
        last_raw = ""
        last_errors = []
        while attempt < self.max_attempts:
            raw_output = self.llm_client.generate_response(prompt=prompt,
                                                max_new_tokens=self.config.max_new_tokens,
                                                do_sample=self.config.do_sample)
            last_raw = raw_output
            result = self.validator.validate(raw_output, paper_id=paper_id)
            if result.is_valid:
                assert result.obj is not None
                accepted = self._accept_policy(judge_obj=result.obj)
                return {
                "id": paper_id,
                "judge": result.obj,
                "accepted": accepted,
                "errors": [],
                "metrics": result.metrics,
            }

            last_errors = result.errors
            attempt += 1
            needs_reeval = any(
                e.startswith("Invalid or missing JSON") or e.startswith("DEGENERATE:")
                for e in last_errors
            )

            if needs_reeval:
                reason = ";".join(last_errors[:3])
                prompt = self.reeval_prompt_builder.build(
                    paper_id=paper_id,
                    abstract=abstract,
                    explanation=explanation[:2000],
                    reason=reason
                )
            else:
                prompt = self.repair_prompt_builder.build(
                    paper_id=paper_id,
                    original_prompt=original_prompt,
                    errors=last_errors,
                    bad_output=raw_output,
                )
        return {
            "id": paper_id,
            "judge": None,
            "accepted": False,
            "errors": last_errors,
            "last_raw": last_raw,
            "metrics": {},
        }
        
    def _accept_policy(self, judge_obj: Dict[str, Any]) -> bool:
        
        CANONICAL_TAGS = {"hallucination", "insufficient_coverage","style_drift", "overlap", "truncation", "non_json", "generic_filler"}
        hard_reject = {"non_json", "truncation"}
        error_tags = []
        
        if not judge_obj:
            return False
        scores = judge_obj.get("scores", {})
        raw_tags = judge_obj.get("error_tags", [])

        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]

        error_tags = []
        for tag in raw_tags:
            if not isinstance(tag, str):
                continue
            base = tag.split(":", 1)[0].strip().lower()
            if base in CANONICAL_TAGS:
                error_tags.append(base)

        if any(t in hard_reject for t in error_tags):
            return False
        faith = int(scores.get("faithfulness", 0) or 0)
        overall = int(scores.get("overall", 0) or 0)

        if "hallucination" in error_tags and faith <= 2:
            return False

        # minimal: use your config thresholds (overall>=60 instead of 70)
        return (faith >= self.config.min_faithfulness_to_keep) and (overall >= self.config.min_overall_to_keep)
    
        

    def judge_many(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            paper_id = item["id"]
            abstract = item["abstract"]
            explanation = item["explanation"]
            result = self.judge_one(paper_id=paper_id, abstract=abstract, explanation=explanation)
            results.append(result)
        return results
    
class JudgeJsonExtractor:
    def __init__(self, raw_text: str, paper_id: str):
        self.raw_text = raw_text.strip()
        self.paper_id = paper_id
    
    def iter_json_object_strings(self) -> Iterator[str]:
        """
        Yields balanced {...} substrings by brace counting.
        Note: does not attempt to ignore braces inside quoted strings.
        """
        text = self.raw_text
        n = len(text)
        i = 0

        while i < n:
            start = text.find("{", i)
            if start == -1:
                return

            brace = 0
            for j in range(start, n):
                ch = text[j]
                if ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0:
                        yield text[start : j + 1]
                        i = j + 1  # continue scanning after this object
                        break
            else:
                # Ran off the end without closing braces
                return

    def extract_obj(self) -> Optional[Dict[str, Any]]:
        """
        Returns the first JSON object that:
        - parses as JSON
        - is a dict
        - has id == paper_id
        """
        text = self.raw_text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and obj.get("id") == self.paper_id:
                    return obj
            except json.JSONDecodeError:
                pass
        for json_str in self.iter_json_object_strings():
            try:
                obj = json.loads(json_str)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("id") == self.paper_id:
                return obj
        return None