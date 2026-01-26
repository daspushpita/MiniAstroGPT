from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, sqlite3
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
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
    max_new_tokens: int = 256
    do_sample: bool = False
    
    min_faithfulness_to_keep: int = 3
    
    REQUIRED_TOP_KEYS = {"id", "scores", "error_tags", "rationale", "rewrite_hint"}
    REQUIRED_SCORE_KEYS = {"faithfulness", "clarity", "jargon", "structure", "uncertainty", "overall"}


class JudgePromptBuilder:
    """Builds the evaluation prompt given abstract + explanation."""
    
    def build(self, *, paper_id: str, abstract: str, explanation: str) -> str:
        return f"""
You are a strict evaluator for scientific-to-layman explanations.
TASK:
Given:
(1) {paper_id}
(2) {abstract} (source of truth)
(3) a teacher-written explanation

Provide the scores for each of the following categories:
- Faithfulness(0-5): How accurately does the explanation reflect the abstract?
- Clarity(0-5): How clearly is the explanation written?
- Jargon(0-5): How much technical jargon is used?
- Structure(0-5): How well is the explanation organized?
- Uncertainty(0-5): How much uncertainty is expressed in the explanation?
- Overall(0-100): A general assessment of the explanation.


If the explanation contains claims NOT supported by the abstract, set:
- error_tags include "hallucination"
- faithfulness <= 2

OUTPUT RULES (VERY IMPORTANT):
- Output ONLY a single JSON object.
- No extra keys.
- All scores must be integers.
- error_tags must be a list of strings.
- rationale must be 1 sentence.
- rewrite_hint must be 1 concrete actionable sentence.

JSON SCHEMA:
{{
"id": "<paper_id>",
"scores": {{
    "faithfulness": 0-5,
    "clarity": 0-5,
    "jargon": 0-5,
    "structure": 0-5,
    "uncertainty": 0-5,
    "overall": 0-100
  }},
  "error_tags": ["..."],
  "rationale": "...",
  "rewrite_hint": "..."
}}

paper_id: {paper_id}
abstract: {abstract}
teacher_explanation: {explanation}
""".strip()


class JudgeValidator:
    """Validates and parses judge JSON output."""
    def __init__(self, model_name):
        
        self.my_config = JudgeConfig()
        self.my_validator = Validation_Regeneration(teacher_model=model_name,
                                                    max_attempts=3,
                                                    prompt_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/teacher_prompt_v1.txt"),
                                                    raise_on_fail=True,
                                                    context_check_mode="off",
                                                    judge_model="judge")

    def validate(self, raw_text: str, paper_id: str):
        errors, metrics = [], {}
        text = (raw_text or "").strip()
        json_text = self.my_validator._attempt_autofix_to_json(text, paper_id, mode="judge")        
        if not json_text:
            return JudgeResult(is_valid=False, errors=["Empty output from teacher"], obj=None, metrics=metrics)
        
        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError:
            return JudgeResult(False, ["Invalid JSON"], None, metrics)

        valid, msg = self._check_required_keys(obj)
        if not valid:
            errors.append(msg)
        valid, msg = self._check_id(obj.get("id", ""), paper_id=paper_id)
        if not valid:
            errors.append(msg)
        valid, msg = self._check_scores(obj.get("scores", {}))
        scores = obj.get("scores", {})
        metrics["faithfulness"] = scores.get("faithfulness")
        metrics["overall"] = scores.get("overall")
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

        return JudgeResult(is_valid=(len(errors) == 0), errors=errors, obj=obj, metrics=metrics)

    def _check_required_keys(self, obj: Dict[str, Any]) -> List[Any]:
        req_keys = JudgeConfig.REQUIRED_TOP_KEYS
        missing =[k for k in req_keys if k not in obj]
        if missing:
            return [False, f"Missing required keys: {missing}"]
        return [True, ""]
    
    def _check_id(self, extracted_id: Any, *, paper_id: str) -> List[Any]:
        if extracted_id != paper_id:
            return [False, f"ID mismatch: expected {paper_id}, got {extracted_id}"]
        return [True, ""]

    def _check_scores(self, scores: Dict[str, Any]) -> List[Any]:
        score_keys = JudgeConfig.REQUIRED_SCORE_KEYS
        for key in score_keys:
            if key not in scores:
                return [False, f"Missing score for: {key}"]
            value = scores[key]
            if not isinstance(value, int):
                return [False, f"Score for {key} is not an interger"]
            if key == "overall":
                if not (0 <= value <= 100):
                    return [False, f"Overall score {value} out of range (0-100)"]
            else:
                if not (0 <= value <= 5):
                    return [False, f"Score for {key} {value} out of range (0-5)"]
        
        return [True, ""]
    
    def _check_error_tags(self, error_tags: Any) -> List[Any]:
        if not isinstance(error_tags, list):
            return [False, "error_tags is not a list"]
        for tag in error_tags:
            if not isinstance(tag, str):
                return [False, f"error_tag {tag} is not a string"]
        return [True, ""]
    
    def _check_rationale(self, rationale: Any) -> List[Any]:
        if not isinstance(rationale, str):
            return [False, "rationale is not a string"]
        text = rationale.strip()
        if ((len(text.split(".")) > 2) or (len(text.split("!")) > 2) or (len(text.split("?")) > 2)):
            return [False, "Rationale is more than 1 sentence"]
        return [True, ""]
    
    def _check_rewrite_hint(self, rewrite_hint: Any) -> List[Any]:
        if not isinstance(rewrite_hint, str):
            return [False, "rewrite_hint is not a string"]
        text = rewrite_hint.strip()
        if ((len(text.split(".")) > 2) or (len(text.split("!")) > 2) or (len(text.split("?")) > 2)):
            return [False, "rewrite_hint is more than 1 sentence"]
        return [True, ""]

class JudgeRepairPromptBuilder:
    """Builds a repair prompt when judge output is invalid."""
    
    def build(self, *, paper_id: str, errors: List[str], bad_output: str) -> str:
        
        err_block = "\n".join(f"- {e}" for e in errors[:8])  # cap to avoid huge prompts
        return f"""
Your previous output was invalid.

Fix these issues:
{err_block}

Rules:
- Output ONLY a single JSON object (no markdown, no commentary).
- Use EXACT keys: id, scores, error_tags, rationale, rewrite_hint
- All scores must be integers.
- rationale = 1 sentence; rewrite_hint = 1 actionable sentence.
- id must equal: {paper_id}

Bad output:
{bad_output}
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
                max_attempts: int = 3):
        
        self.llm_client = llm_client
        self.config = config or JudgeConfig()
        self.prompt_builder = prompt_builder or JudgePromptBuilder()
        self.validator = validator or JudgeValidator(model_name=self.llm_client.model_id)
        self.repair_prompt_builder = repair_prompt_builder or JudgeRepairPromptBuilder()
        self.max_attempts = max_attempts

    def judge_one(self, *, paper_id: str, abstract: str, explanation: str) -> Dict[str, Any]:
        
        prompt = self.prompt_builder.build(paper_id=paper_id, abstract=abstract, explanation=explanation)
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
            prompt = self.repair_prompt_builder.build(paper_id=paper_id, errors=result.errors, bad_output=raw_output)
        return {
            "id": paper_id,
            "judge": None,
            "accepted": False,
            "errors": last_errors,
            "last_raw": last_raw,
            "metrics": {},
        }
        
    def _accept_policy(self, judge_obj: Dict[str, Any]) -> bool:
        if not judge_obj:
            return False
        scores = judge_obj.get("scores", {})
        error_tags = judge_obj.get("error_tags", [])
        if "hallucination" in error_tags:
            return False
        if scores.get("faithfulness", 0) < self.config.min_faithfulness_to_keep:
            return False
        return True            

    def judge_many(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            paper_id = item["id"]
            abstract = item["abstract"]
            explanation = item["explanation"]
            result = self.judge_one(paper_id=paper_id, abstract=abstract, explanation=explanation)
            results.append(result)
        return results