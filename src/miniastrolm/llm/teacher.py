from __future__ import annotations

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, sqlite3
from collections import Counter
from typing import Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

# DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def check_mps_availability() -> bool:
    """checks if MPS (Metal Performance Shaders) is available for PyTorch
    """
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

class Llama_Teacher:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                device: str | None = None,
                torch_dtype: torch.dtype = torch.float16,):
        
        """Initializes the Llama Teacher model and tokenizer.
        Mac GPU path uses MPS.
        """
        if device is None:
            device = "mps" if check_mps_availability() else "cpu"
        
        self.device = device
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        
        if self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch_dtype,
                device_map={"": "mps"},
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            
        self.model.eval()

    def generate_response(self,
                        prompt: str,
                        max_new_tokens: int = 800,
                        # temperature : float = 0.7,
                        # top_p: float = 0.9,
                        repetition_penalty: float = 1.1,
                        do_sample=False) -> str:
        """Generates responses from a given prompt

        Args:
            tokenizer (_type_): The tokenizer instance
            model (_type_): The model instance
            prompt (str): The input prompt string
        """

        # 1) tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 2) move to device (MPS or CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # keep this length so we can slice out only the new tokens later
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # temperature=temperature,
                # top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            )
        # 3. Decode the output tokens to text
        output_text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return output_text.strip()
    
    def generate_response_chat(self,
                        system_prompt: str,
                        user_prompt: str,
                        max_new_tokens: int = 800,
                        # temperature : float = 0.7,
                        # top_p: float = 0.9,
                        repetition_penalty: float = 1.1,
                        do_sample=False) -> str:
        """Generates responses from a given prompt

        Args:
            tokenizer (_type_): The tokenizer instance
            model (_type_): The model instance
            prompt (str): The input prompt string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # 1) tokenize
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 2) move to device (MPS or CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # keep this length so we can slice out only the new tokens later
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # temperature=temperature,
                # top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            )
        # 3. Decode the output tokens to text
        output_text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return output_text.strip()
    
class Validation_Regeneration:
    def __init__(self, teacher_model: Llama_Teacher, max_attempts: int = 3, 
                prompt_path : Path | None = None, 
                raise_on_fail: bool = True, 
                context_min_coverage: float = 0.6, 
                context_check_mode: str = "off", 
                judge_model=None):
        
        self.teacher_model = teacher_model
        self.max_attempts = max_attempts
        #self.min_chars = 800
        self.min_chars = 250
        self.banned_phrases = [
            "we present", "we show", "we find", "we propose", "we investigate",
            "in this paper", "this paper", "we study", "we explore",
            "we report", "we demonstrate",
        ]
        self.raise_on_fail = raise_on_fail
        self.context_min_coverage = context_min_coverage
        self.context_check_mode = context_check_mode
        self.judge_model = judge_model or teacher_model

        
        if prompt_path is None:
            raise ValueError("Base Teacher prompt path must be provided.")
        self.base_prompt_template = prompt_path.read_text(encoding="utf-8").strip()

    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """
        Normalize arXiv IDs so equivalent formats compare equal.

        Examples normalized to: 1801.00386v1
        - http://arxiv.org/abs/1801.00386v1
        - https://arxiv.org/abs/1801/00386v1
        - arxiv.org/abs/1801.00386v1
        """
        s = arxiv_id.strip().lower()

        # Remove URL prefix if present
        if "arxiv.org/abs/" in s:
            s = s.split("arxiv.org/abs/")[-1]

        # Normalize slash-based IDs to dot-based
        # 1801/00386v1 → 1801.00386v1
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2 and parts[0].isdigit():
                s = parts[0] + "." + parts[1]

        return s


    def _base_prompt(self, paper_id: str, abstract: str) -> tuple[str, str]:
        """
        A function for generating a base prompt.
        
        :param self: Description
        :param paper_id: Description
        :type paper_id: str
        :param abstract: Description
        :type abstract: str
        :return: Description
        :rtype: tuple[str, str]
        """
        
        system_prompt = self.base_prompt_template.format(paper_id=paper_id, abstract=abstract)

        key_phrases = self._extract_key_phrases(abs_str=abstract, k=8)
        must_cover = "\n".join(f"- {p}" for p in key_phrases)

        system_prompt = (
            system_prompt
            + "\n\nMUST-COVER CONCEPTS (paraphrase, do not quote):\n"
            + must_cover
            + "\n\nIf any must-cover concept is missing, the output fails."
        )

        user_prompt = (
                f"paper_id: {paper_id}\n\n"
                f"ABSTRACT:\n{abstract}\n\n"
                f"Return ONLY the JSON object."
            )
        return system_prompt, user_prompt
        
    def _extract_json_object(self, text: str) -> str | None:
        """Extracts the first JSON object from the given text.
            This will fail if here are "{}" inside the string value of "explanation"...
            Update this function later if this becomes an issue
        """
        start = text.find("{")
        if start == -1:
            return None
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]
        return None
    
    def _escape_json_string(self, s: str) -> str:
        """
        Convert arbitrary text into a safe JSON string body:
        - escapes backslashes and double-quotes
        - converts real newlines/tabs to JSON escapes
        - removes carriage returns
        """
        s = s.replace("\r", "")
        s = s.replace("\\", "\\\\")
        s = s.replace('"', '\\"')
        s = s.replace("\t", "\\t")
        s = s.replace("\n", "\\n")
        return s
    
    def _attempt_autofix_to_json(self, raw_text: str, paper_id: str, mode: str = "teacher",) -> str | None:
        """
        Attempt to repair a near-JSON output where the explanation value
        exists but is not wrapped in quotes.

        Returns a valid JSON string if successful, otherwise None.
        """

        if mode == "teacher":
            required_key = "explanation"
        elif mode == "judge":
            required_key = "verdict"
        else:
            return None

        # 1. Normalize input
        text = (raw_text or "").strip()

        existing = self._extract_json_object(text)
        if existing is not None:
            return existing
        
        if f'"{required_key}"' not in text:
            return None
        
        # Judge "autofix" = extract first JSON object only (do not invent/overwrite fields)
        if mode == "judge":
            extracted = self._extract_json_object(text)
            return extracted  # may be None

        # 4. Extract id if present; otherwise fall back to provided paper_id
        m_id = re.search(r'"id"\s*:\s*"([^"]+)"', text)
        extracted_id = m_id.group(1) if m_id else paper_id

        # 5. Locate where the explanation value should begin
        m_exp = re.search(r'"explanation"\s*:\s*', text)
        if not m_exp:
            return None

        explanation_body = text[m_exp.end():].strip()

        # 6. If explanation already starts with a quote, this is not our failure mode
        after_l = explanation_body.lstrip()
        if after_l.startswith('"'):
            return None

        # 7. Remove a trailing closing brace if present
        if explanation_body.endswith("}"):
            explanation_body = explanation_body[:-1].strip()
        
        # 8) Optional: if model accidentally left a trailing quote at the end, trim it
        if explanation_body.endswith('"'):
            explanation_body = explanation_body[:-1].rstrip()

        # 9) Optional: if model accidentally starts body with a comma/newline, clean it
        body = explanation_body.lstrip(",").lstrip()

        if not body:
            return None
    
        # 10. Rebuild a strict JSON object safely
        fixed_obj = {
            "id": extracted_id,
            "explanation": body,
        }

        # 11. Serialize using json.dumps to guarantee valid JSON escaping
        return json.dumps(fixed_obj, ensure_ascii=False)


    def _validate_response(self, explanation: str, paper_id: str, abstract: str) ->  Tuple[bool, List[str], Dict[str, Any] | None]:
        errors: List[str] = []
        text = (explanation or "").strip()
        json_text = self._attempt_autofix_to_json(text, paper_id)
        
        if json_text is None:
            snippet = text[:200].replace("\n", "\\n")
            return False, [f"No JSON object found. Snippet: {snippet!r}"], None
        
        try:
            obj = json.loads(json_text)
        except Exception as e:
            snippet = json_text[:200].replace("\n", "\\n")
            return False, [f"Invalid JSON: {e}. Snippet: {snippet!r}"], None
        
        # minimal field checks-----------------------------------------------------------------
        if not isinstance(obj, dict):
            errors.append("Response is not a JSON object")
            return False, errors, None

        if "id" not in obj or "explanation" not in obj:
            return False, ['JSON must contain keys: "id" and "explanation".'], None
    
        if self._normalize_arxiv_id(obj["id"]) != self._normalize_arxiv_id(paper_id):
            return False, [f"Paper ID mismatch: expected {paper_id}, got {obj['id']}"], None

        #----------------------------------------------------------------------------------------
        explanation_text = (obj.get("explanation") or "").strip()
        
        if explanation_text == "":
            errors.append("Empty response")
            return False, errors, None

        if len(explanation_text) < self.min_chars:
            errors.append(f"Response too short: {len(explanation_text)} characters")
            
        low = explanation_text.lower()
        for phrase in self.banned_phrases:
            if phrase in low:
                errors.append(f"Banned phrase found: '{phrase}'")
        
        if self._has_long_overlap(explanation_text, abstract, n=12):
            errors.append("Likely copied from abstract (12+ word overlap detected).")

        is_valid = (len(errors) == 0)
        return is_valid, errors, obj
    
    def _has_long_overlap(self, explanation: str, abstract: str, n: int = 12) -> bool:
        explanation_words = re.findall(r'\w+', explanation.lower())
        abstract_words = re.findall(r'\w+', abstract.lower())
        
        expl_len = len(explanation_words)
        abs_len = len(abstract_words)
        
        abs_ngrams = {tuple(abstract_words[j:j+n]) for j in range(abs_len - n + 1)}
        for i in range(expl_len - n + 1):
            if tuple(explanation_words[i:i+n]) in abs_ngrams:
                return True
        return False
    
    def _repair_prompt(self, *, paper_id: str, abstract: str, bad_output: str, errors: List[str], attempt: int) -> str:
        
        system_prompt, user_prompt = self._base_prompt(paper_id, abstract)
        errors_bullets = "\n".join(f"- {e}" for e in errors)
        repair = f"""Your previous explanation FAILED validation.

PAPER ID: "{paper_id}"

VALIDATION ERRORS:
{errors_bullets}

HARD REQUIREMENTS:
- Output ONLY valid JSON. No markdown. No commentary. No extra text.
- Top-level output MUST be a single JSON object (not an array).
- The JSON object MUST have keys: "id", "explanation" (additional keys are allowed).
- "id" MUST equal: "{paper_id}"
- "explanation" MUST be at least {self.min_chars} characters.
- Must NOT use academic-reporting phrases (e.g., "we present", "in this paper", etc.).
- Must NOT start with: “The paper…”, “This paper…”, “The authors…”, “This study…”
- Must NOT mention the paper, authors, or “the abstract”.
- Write as a direct popular-science explanation (magazine tone).
- Begin by explaining the underlying phenomena (definitions first), then mechanisms, then implications.
- No bullets/headings in the explanation text.
- Must be rewritten from scratch and not copy phrases from the abstract.
- Define key technical terms the first time they appear (1 short sentence each).
- Examples of “key terms”: the main physical ingredients, objects, or mechanisms in the abstract (e.g., a kind of matter/field, a signal being measured, an effect/process).
- Do not include ```json fences or any text before/after the JSON.
- Do not reuse any sequence of 12 consecutive words from the abstract.
- Change sentence structure. Use different wording. Avoid reusing multi-word phrases.

If the abstract contains a modifier that changes meaning (e.g., “cold”, “primordial”, “frequency-dependent”), briefly explain what that modifier implies.
ABSTRACT:
{abstract}

YOUR INVALID OUTPUT (for reference only — do NOT reuse wording):
{bad_output}

Now output ONLY this JSON object, starting with '{{' and ending with '}}':
{{"id":"{paper_id}","explanation":"..."}}
"""

        user_prompt = user_prompt + "\n\n" + repair
        return system_prompt, user_prompt

    def _context_check_interactive(self, paper_id: str, 
                        abstract: str, 
                        explanation: str,
                        max_phrases: int = 10) -> Tuple[bool, float, List[str], Dict[str, Any] | None]:

        phrases = self._extract_key_phrases(abs_str=abstract, k=max_phrases)
        
        #Normalize the explanation
        exp_norm = explanation.lower()
        exp_norm = re.sub(r"[^a-z0-9\s]", " ", exp_norm)
        exp_norm = re.sub(r"\s+", " ", exp_norm).strip()
        
        matched = 0
        missing: List[str] = []
        
        for my_phrase in phrases:
            if my_phrase in exp_norm:
                matched += 1
            else:
                missing.append(my_phrase)
                
        
        total = len(phrases)
        score = matched / total if total > 0 else 0.0
        
        ok = score > self.context_min_coverage
        return ok, score, missing, None
        

    def _extract_key_phrases(self, abs_str: str, k: int) -> List[str]:
        
        """
        Extracting key phrases occuring in the abstract
        
        :param abs_str: Description
        :type abs_str: str
        :param k: Description
        :type k: int
        :return: Description
        :rtype: List[str]
        """
        _STOPWORDS = {
        "a","an","the","and","or","but","if","then","else","so","because","as","of","to","in","on","at","by","for",
        "with","without","from","into","over","under","between","within","through","during",
        "is","are","was","were","be","been","being","have","has","had","do","does","did",
        "this","that","these","those","we","they","their","them","it","its","our","us","you","your",
        "can","could","may","might","must","should","would","will",
        "such","than","also","however","there","here","more","most","less","least",
        "using","use","used","via","based","show","shows","shown","present","study","results"
        }
        words = abs_str.lower()
        words = re.sub(r"[^a-z0-9]+", " ", words)
        words = words.split()
        
        if len(words) == 0:
            raise ValueError("Empty abstract with no words")
        if k == 0:
            return []
        if k < 0:
            raise ValueError("k must be non-negative")
        
        filtered_words = [w for w in words if w not in _STOPWORDS]
        
        counts: Counter[str] = Counter()
        for n in (2, 3, 4):
            for i in range(len(filtered_words) - n + 1):
                gram_tokens = filtered_words[i:i+n]
                phrase = " ".join(gram_tokens)
                counts[phrase] += 1
                
        if not counts:
            return []
        
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        results: List[str] = []
        for phrase, _freq in counts:
            if any(phrase in kept or kept in phrase for kept in results):
                continue
            results.append(phrase)
            if len(results) >= k:
                break
        return results
    
    def _judge_prompt(self, *, paper_id: str, abstract: str, explanation: str) -> str:
        return f"""
You are a strict scientific fact-checker.
Task: Compare the EXPLANATION against the ABSTRACT.
- Penalize missing key points (major omissions).
- Penalize hallucinations (claims not supported by abstract).
- Do NOT require extra details beyond abstract.
- overall_score should reflect faithfulness and coverage jointly
(high only if both coverage is good and hallucinations are low).
- verdict MUST be "pass" only if overall_score >= {self.context_min_coverage}.
- Do not include ```json fences or any text before/after the JSON.
Return ONLY valid JSON:
{{
    "id": "{paper_id}",
    "verdict": "pass" | "fail",
    "coverage_score": 0.0-1.0,
    "hallucination_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "missing_key_points": ["..."],
    "unsupported_claims": ["..."],
    "notes": "short"
}}

ABSTRACT:
{abstract}

EXPLANATION:
{explanation}
""".strip()


        
    def _context_check_llm(self, paper_id: str, 
                        abstract: str, 
                        explanation: str,
                        max_phrases: int = 10) -> Tuple[bool, float, List[str], Dict[str, Any] | None]:
        
        prompt = self._judge_prompt(paper_id=paper_id, abstract=abstract, explanation=explanation)
        raw_response = self.judge_model.generate_response(prompt=prompt, max_new_tokens=512, do_sample=False)
        # json_txt = self._extract_json_object(raw_response)
        json_txt = self._attempt_autofix_to_json(raw_response, paper_id, mode="judge")
        if json_txt is None:
            return False, 0.0, ["Judge returned no JSON"], None
        try:
            json_obj = json.loads(json_txt)
        except Exception as e:
            snippet = json_txt[:200].replace("\n", "\\n")
            return False, 0.0, [f"Judge returned invalid JSON: {e}"], None
        
        # minimal field checks
        # - id match
        if self._normalize_arxiv_id(json_obj.get("id")) != self._normalize_arxiv_id(paper_id):
            return False, 0.0, [f"Paper ID mismatch: expected {paper_id}, got {json_obj.get('id')}"], None

        # - overall_score exists
        if "overall_score" not in json_obj:
            return False, 0.0, [f"Judge returned no overall_score"], None
        # - verdict exists
        # (keep strict; fail closed)
        if "verdict" not in json_obj:
            return False, 0.0, [f"Judge returned no verdict"], None
        verdict = json_obj.get("verdict")
        
        if verdict not in ("pass", "fail"):
            return False, 0.0, [f"Judge returned invalid verdict: {verdict}"], None
        
        score = float(json_obj.get("overall_score", 0.0))

        problems: List[str] = []
        problems += json_obj.get("unsupported_claims", [])
        problems += json_obj.get("missing_key_points", [])
        
        ok = (verdict == "pass") and (score > self.context_min_coverage)
        return ok, score, problems, json_obj
        
    def _context_repair_prompt(self, *, paper_id: str, abstract: str, bad_output: str,
                            missing: List[str], ctx_score: float, attempt: int) -> str:
        
        system_prompt, user_prompt = self._base_prompt(paper_id, abstract)
        missing_bullets = "\n".join(f"- {m}" for m in missing)
        
        ctx =  f"""
Your explanation passed formatting but failed context coverage.
PAPER ID: "{paper_id}"
CONTEXT COVERAGE SCORE: {ctx_score:.2f}
MISSING CONCEPTS TO COVER (paraphrase, do not quote):
{missing_bullets}
Rewrite the explanation to include the missing concepts (only if supported by the abstract).
Do not add any new claims not grounded in the abstract.
Keep the same popular-science style rules as the base prompt.
Define key technical terms the first time they appear.

Return ONLY JSON:
{{"id":"{paper_id}","explanation":"..."}}

ABSTRACT:
{abstract}

PREVIOUS EXPLANATION (do not copy wording):
{bad_output}
""".strip()
        user_prompt = user_prompt + "\n\n" + ctx
        return system_prompt, user_prompt   

    def _run_context_check(self, *, paper_id: str, abstract: str, explanation_text: str) -> Tuple[bool, float, List[str], Dict[str, Any] | None]:
        
        if self.context_check_mode == "off":
            return True, 1.0, [], None
        elif self.context_check_mode == "interactive":
            return self._context_check_interactive(paper_id=paper_id, abstract=abstract, explanation=explanation_text)
        elif self.context_check_mode == "llm":
            return self._context_check_llm(paper_id=paper_id, abstract=abstract, explanation=explanation_text)
        else:
            raise ValueError("Context check mode not recognized")

    def generate_item(self, paper_id: str, abstract: str) -> Dict[str, Any]:
        attempt = 0
        last_response = ""
        last_errors: List[str] = []
        system_prompt, user_prompt = self._base_prompt(paper_id, abstract)
        
        while attempt < self.max_attempts:
            system_prompt, user_prompt = self._base_prompt(paper_id, abstract)
            response = self.teacher_model.generate_response_chat(system_prompt=system_prompt, user_prompt=user_prompt, max_new_tokens=800, do_sample=False)
            last_response = response
            is_valid, errors, obj = self._validate_response(response, paper_id, abstract)

            if not is_valid:
                attempt += 1
                last_errors = errors
                system_prompt, user_prompt = self._repair_prompt(
                    paper_id=paper_id,
                    abstract=abstract,
                    bad_output=response,
                    errors=errors,
                    attempt=attempt,
                )
                continue

            if not isinstance(obj, dict):
                attempt += 1
                last_errors = ["Unexpected: obj is not dict after validation."]
                system_prompt, user_prompt = self._repair_prompt(
                    paper_id=paper_id,
                    abstract=abstract,
                    bad_output=response,
                    errors=errors,
                    attempt=attempt,
                )
                continue

            explanation_text = (obj.get("explanation") or "").strip()
            ok_ctx, score_ctx, problems_ctx, obj_ctx = self._run_context_check(paper_id=paper_id, abstract=abstract, explanation_text=explanation_text)
            if not ok_ctx:
                last_errors = list(errors) + list(problems_ctx)
                attempt += 1
                system_prompt, user_prompt = self._context_repair_prompt(
                        paper_id=paper_id,
                        abstract=abstract,
                        bad_output=response,
                        missing=problems_ctx,
                        ctx_score=score_ctx,
                        attempt=attempt
                    )
                continue

            obj["accepted"] = True
            obj["attempts"] = attempt + 1
            obj["validation_errors"] = []
            obj["context_score"] = score_ctx
            obj["context_problems"] = problems_ctx
            obj["context_judge"] = obj_ctx
            return obj


        failure_obj = {
            "id": paper_id,
            "explanation": "",
            "accepted": False,
            "attempts": attempt,
            "validation_errors": last_errors,
            "last_response": last_response,
        }
        
        if self.raise_on_fail:
            raise ValueError(
                f"Failed to generate valid response after {self.max_attempts} attempts.\n"
                f"Last response: {last_response}\n"
                f"Errors: {last_errors}"
            )
        else:
            return failure_obj
        
        
class Teacher_Data_Pipeline:
    def __init__(self, database_path, output_path, prompt_path, 
                batch_size = 100, max_attempts = 3,
                raise_on_fail = False,
                log_skips=False,        
                print_every=1,):
        
        self.database_path = database_path
        self.output_path = output_path
        self.prompt_path = prompt_path
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.raise_on_fail = raise_on_fail
        self.log_skips = log_skips
        self.print_every = print_every
        
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.train_path = self.output_path / "train.jsonl"
        self.fail_path = self.output_path / "failures.jsonl"
        self.processed_ids_path = self.output_path / "processed_ids.txt"
        self.cursor_path = self.output_path / "cursor_last_id.txt"

        self.run_log_path = self.output_path / "run_log.jsonl"
        self.run_summary_path = self.output_path / "run_summary.json"
        
        self._processed = set()
        self._last_id = None
        self._total_eligible = None

    def setup(self):
        self._processed = self._load_processed_ids(self.processed_ids_path)
        self._last_id = self._load_last_id(self.cursor_path)
        self.teacher = Llama_Teacher()
        self.validator = Validation_Regeneration(
                    teacher_model=self.teacher,
                    max_attempts=self.max_attempts,
                    prompt_path=self.prompt_path,
                    raise_on_fail=self.raise_on_fail,
                    context_check_mode="off",
                    judge_model=None,
                )
        
    def _load_processed_ids(self, path):
        if not Path(path).exists():
            return set()
        ids = set()
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    ids.add(s)
        return ids
        
    def _load_last_id(self, path):
        if not Path(path).exists():
            return None
        s = Path(path).read_text(encoding="utf-8").strip()
        return s or None
    
    # ----------------------------
    # NEW: Progress tracking helpers
    # ----------------------------

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_summary(self, summary: dict) -> None:
        # overwrite summary each time (cheap + convenient)
        self.run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log_event(self, *, status: str, paper_id: str, **extra) -> None:
        event = {
            "ts": self._now_iso(),
            "status": status,       # "ok" | "fail" | "skip"
            "id": paper_id,
            "last_id": self._last_id,
            **extra,
        }
        self._append_jsonl(self.run_log_path, event)

    def _count_total_eligible(self, conn) -> int:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM papers WHERE abstract_clean IS NOT NULL")
        total = int(cur.fetchone()[0])
        cur.close()
        return total
    
    def process_one(self, paper_id, title, abstract):
        
        try:
            output = self.validator.generate_item(
                                    paper_id= paper_id,
                                    abstract = abstract)
            payload = {
                "id": paper_id,
                "title": title,
                "input": abstract,
                "output": output,  # obj should already be structured dict
                "meta": {
                    "teacher_model": getattr(self.teacher, "model_id", None),
                    "max_attempts": self.max_attempts,
                },
            }
            return True, payload
        except Exception as e:
            fail = {
                "id": paper_id,
                "title": title,
                "input": abstract,
                "error": repr(e),
            }
            return False, fail        

    def fetch_batch(self, conn):

        cursor = conn.cursor()
        if self._last_id is None:
            cursor.execute("""SELECT id, title_clean, abstract_clean FROM papers
                        WHERE abstract_clean IS NOT NULL
                        ORDER BY id
                        LIMIT ?""", (self.batch_size,))
        else:
            cursor.execute(
                """
                SELECT id, title_clean, abstract_clean
                FROM papers
                WHERE abstract_clean IS NOT NULL
                  AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (self._last_id, self.batch_size),)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def mark_processed(self, paper_id: str):
        with self.processed_ids_path.open("a", encoding="utf-8") as f:
            f.write(f"{paper_id}\n")
        self._processed.add(paper_id)
        
        self._last_id = paper_id
        self.cursor_path.write_text(paper_id, encoding="utf-8")
        
    def _append_jsonl(self, path, payload):
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        
    def run(self):
        self.setup()
        assert self.teacher is not None and self.validator is not None

        processed_count = 0
        success_count = 0
        fail_count = 0
        
        skip_count = 0

        conn = sqlite3.connect(self.database_path)
        
        try:
            self._total_eligible = self._count_total_eligible(conn)

            # Write initial summary
            self._write_summary(
                {
                    "started_ts": self._now_iso(),
                    "total_eligible": self._total_eligible,
                    "processed": processed_count,
                    "success": success_count,
                    "fail": fail_count,
                    "skipped": skip_count,
                    "last_id": self._last_id,
                }
            )
            while True:
                rows = self.fetch_batch(conn)
                if not rows:
                    break
                for paper_id, title_clean, abstract_clean in tqdm(
                    rows,
                    desc="Teacher pipeline",
                    unit="paper",
                    leave=False,
                ):
                    if not abstract_clean or not str(abstract_clean).strip():
                        fail_payload = {
                            "id": paper_id,
                            "title": title_clean,
                            "input": abstract_clean,
                            "error": "Empty abstract_clean",
                        }
                        self._append_jsonl(self.fail_path, fail_payload)
                        fail_count += 1
                        processed_count += 1
                        self.mark_processed(paper_id)
                        self._log_event(status="fail", paper_id=paper_id, reason="empty_abstract")
                        continue
                        
                    if paper_id in self._processed:
                        self._last_id = paper_id
                        self.cursor_path.write_text(paper_id, encoding="utf-8")
                        skip_count += 1
                        if self.log_skips:
                            self._log_event(status="skip", paper_id=paper_id)
                        continue

                    ok, payload = self.process_one(paper_id, title_clean, abstract_clean)
                    if ok:
                        self._append_jsonl(self.train_path, payload)
                        success_count += 1
                        self._log_event(status="ok", paper_id=paper_id)
                    else:
                        self._append_jsonl(self.fail_path, payload)
                        fail_count += 1
                        self._log_event(status="fail", paper_id=paper_id, error=payload.get("error"))

                    self.mark_processed(paper_id)
                    processed_count += 1
                    if processed_count % max(1, self.print_every) == 0:
                        pct = (
                            (processed_count / self._total_eligible) * 100.0
                            if self._total_eligible
                            else None
                        )
                        self._write_summary(
                            {
                                "updated_ts": self._now_iso(),
                                "total_eligible": self._total_eligible,
                                "processed": processed_count,
                                "success": success_count,
                                "fail": fail_count,
                                "skipped": skip_count,
                                "percent_done": pct,
                                "last_id": self._last_id,
                            }
                        )
                        print(
                            f"[Teacher_Data_Pipeline] processed={processed_count} "
                            f"success={success_count} fail={fail_count} skipped={skip_count} "
                            f"percent={pct:.2f}% last_id={self._last_id}"
                            if pct is not None
                            else f"[Teacher_Data_Pipeline] processed={processed_count} "
                                f"success={success_count} fail={fail_count} skipped={skip_count} "
                                f"last_id={self._last_id}"
                        )

            # Final summary
            pct = (processed_count / self._total_eligible) * 100.0 if self._total_eligible else None
            self._write_summary(
                {
                    "finished_ts": self._now_iso(),
                    "total_eligible": self._total_eligible,
                    "processed": processed_count,
                    "success": success_count,
                    "fail": fail_count,
                    "skipped": skip_count,
                    "percent_done": pct,
                    "last_id": self._last_id,
                }
            )

            print(
                f"[Teacher_Data_Pipeline DONE] processed={processed_count} "
                f"success={success_count} fail={fail_count} skipped={skip_count} "
                f"percent={pct:.2f}% last_id={self._last_id}"
                if pct is not None
                else f"[Teacher_Data_Pipeline DONE] processed={processed_count} "
                    f"success={success_count} fail={fail_count} skipped={skip_count} "
                    f"last_id={self._last_id}"
            )
        finally:
            conn.close()
            
        
