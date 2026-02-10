from __future__ import annotations
import json
from pathlib import Path
import re, sqlite3
from collections import Counter
from typing import Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
import numpy as np

class Validation_Regeneration:
    def __init__(self, teacher_model, my_config,
                prompt_path : Path | None = None,
                raise_on_fail: bool = True, 
                context_check_mode: str = "off", 
                judge_model=None):
        
        self.my_config = my_config
        self.teacher_model = teacher_model
        self.max_attempts = self.my_config.max_attempts
        self.min_chars = self.my_config.min_chars
        self.architecture = self.my_config.architecture
        self.banned_phrases = [
            "we present", "we show", "we find", "we propose", "we investigate",
            "in this paper", "this paper", "we study", "we explore",
            "we report", "we demonstrate",
        ]
        self.raise_on_fail = raise_on_fail
        self.context_min_coverage = self.my_config.context_min_coverage
        self.context_check_mode = context_check_mode
        self.judge_model = judge_model or teacher_model
        self.end_sentinel = "<END_JSON>"
        self._END_SENTINEL_RE = re.compile(r"\s*<END_JSON>\s*$")
        
        if prompt_path is None:
            raise ValueError("Base Teacher prompt path must be provided.")
        self.base_prompt_template = prompt_path.read_text(encoding="utf-8").strip()

        self._END_OK = re.compile(r'[.!?]"?\s*$')  # ends with .,!,? optionally followed by quote

        self._BAD_TAIL = {
            "and","or","because","which","that","with","to","of","as","if","but","so","while","into","from"
        }

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
        
        system_prompt = self.base_prompt_template.format(paper_id=paper_id)

        key_phrases = self._extract_key_phrases(abs_str=abstract, k=8)
        must_cover = "\n".join(f"- {p}" for p in key_phrases)

        system_prompt = (
            system_prompt
            + "\n\nMUST-COVER CONCEPTS (paraphrase, do not quote):\n"
            + must_cover
            + "\n\nIf any must-cover concept is missing, the output fails."
        )

        if self.architecture == "llama_hf":
            user_prompt = (
                    f"paper_id: {paper_id}\n\n"
                    f"ABSTRACT:\n{abstract}\n\n"
                    f"Return ONLY the JSON object."
                )
        else:
        # user_prompt = (
        #     f"paper_id: {paper_id}\n\n"
        #     f"ABSTRACT:\n{abstract}\n\n"
        #     f"Return ONLY one JSON object on a SINGLE LINE, then write {self.end_sentinel} and STOP.\n"
        #     f"Do not write anything after {self.end_sentinel}."
        # )

            user_prompt = (
                f"paper_id: {paper_id}\n\n"
                f"ABSTRACT:\n{abstract}\n\n"
                f"Return ONLY one JSON object on a SINGLE LINE.\n"
                f"After the final '}}', output {self.end_sentinel} and STOP.\n"
                f"Do not write anything after {self.end_sentinel}."
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


        if mode != "teacher":
            return None  # keep teacher-only here; judge has its own path now

        required_key = "explanation"

        # 1) Normalize
        text = (raw_text or "").strip()
        if not text:
            return None

        existing_extracted = self._extract_json_object(text)
        if existing_extracted is not None:
            try:
                json.loads(existing_extracted)
                return existing_extracted
            except json.JSONDecodeError:
                # fall through to targeted repair
                pass
        
        # 3) If it doesn't even mention the key, we can't do this targeted fix
        if f'"{required_key}"' not in text:
            return None

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

        # 7) SAFETY: only apply this repair if explanation appears to be the last field.
        # Heuristic: after the explanation begins, there should NOT be another JSON key pattern like: ,"foo":
        # (We accept whitespace/newlines before the comma.)
        if re.search(r'\n?\s*,\s*"[A-Za-z0-9_ ]+"\s*:', explanation_body):
            return None
    
        # 8) Trim trailing brace if present (common when model forgot quotes)
        body = explanation_body.strip()
        if body.endswith("}"):
            body = body[:-1].rstrip()
        
        # 9) Trim accidental trailing quote
        if body.endswith('"'):
            body = body[:-1].rstrip()

        # 10) Trim leading comma/newline noise
        body = body.lstrip(",").lstrip()
        if not body:
            return None
    
        fixed_obj = {"id": extracted_id, "explanation": body}
        return json.dumps(fixed_obj, ensure_ascii=False)



    def _validate_response(self, explanation: str, paper_id: str, abstract: str) ->  Tuple[bool, List[str], Dict[str, Any] | None]:
        errors: List[str] = []
        text = (explanation or "").strip()
        # If sentinel exists, keep only content before it
        if self.end_sentinel in text:
            text = text.split(self.end_sentinel, 1)[0].strip()
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
        
        if self._has_long_overlap(explanation_text, abstract, n=20):
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
    
    def _repair_prompt(self, *, paper_id: str, abstract: str, bad_output: str, errors: List[str], attempt: int) -> tuple[str, str]:
        
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
        return (system_prompt, user_prompt)

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
        
        sorted_counts: List[Tuple[str, int]] = sorted(
            counts.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        results: List[str] = []
        for phrase, _freq in sorted_counts:
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
        raw_response = self.judge_model.generate_response(prompt=prompt, 
                                                        max_new_tokens=self.my_config.max_new_tokens, 
                                                        do_sample=self.my_config.do_sample, 
                                                        temperature=self.my_config.temperature, 
                                                        top_p=self.my_config.top_p, 
                                                        repetition_penalty=self.my_config.repetition_penalty)
        
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
                            missing: List[str], ctx_score: float, attempt: int) -> tuple[str, str]:
        
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
        return (system_prompt, user_prompt)   

    def _run_context_check(self, *, paper_id: str, abstract: str, explanation_text: str) -> Tuple[bool, float, List[str], Dict[str, Any] | None]:
        
        if self.context_check_mode == "off":
            return True, 1.0, [], None
        elif self.context_check_mode == "interactive":
            return self._context_check_interactive(paper_id=paper_id, abstract=abstract, explanation=explanation_text)
        elif self.context_check_mode == "llm":
            return self._context_check_llm(paper_id=paper_id, abstract=abstract, explanation=explanation_text)
        else:
            raise ValueError("Context check mode not recognized")

    def looks_truncated(self, explanation: str) -> bool:
        text = (explanation or "").strip()

        # too short for your spec => treat as truncated
        if len(text) < self.my_config.min_chars:
            return True

        # must end like a complete sentence (this catches your "seem too small" case)
        if not self._END_OK.search(text):
            return True

        # unbalanced quotes is a strong truncation signal
        if text.count('"') % 2 == 1:
            return True

        return False
        
        
    def generate_item(self, paper_id: str, abstract: str) -> Dict[str, Any]:
        attempt = 0
        last_response = ""
        last_errors: List[str] = []
        did_token_retry = False
        gen_calls = 0
        system_prompt, user_prompt = self._base_prompt(paper_id, abstract)
        while attempt < self.my_config.max_attempts:
            response = self.teacher_model.generate_response_chat(system_prompt=system_prompt, 
                                                                user_prompt=user_prompt, 
                                                                max_new_tokens=self.my_config.max_new_tokens, 
                                                                do_sample=self.my_config.do_sample, 
                                                                temperature=self.my_config.temperature, 
                                                                top_p=self.my_config.top_p, 
                                                                repetition_penalty=self.my_config.repetition_penalty)

            gen_calls += 1
            last_response = response
            
            if (not did_token_retry) and response.lstrip().startswith("{") and ("\"id\"" in response) and ("\"explanation\"" in response):
                if response.count("{") > response.count("}"):  # missing closing brace => truncated JSON
                    did_token_retry = True
                    response = self.teacher_model.generate_response_chat(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_new_tokens=int(self.my_config.max_new_tokens_retry),
                        do_sample=self.my_config.do_sample,
                        temperature=self.my_config.temperature,
                        top_p=self.my_config.top_p,
                        repetition_penalty=self.my_config.repetition_penalty,
                    )
                    gen_calls += 1
                    last_response = response
                    
            is_valid, errors, obj = self._validate_response(response, paper_id, abstract)
            if not is_valid:
                attempt += 1
                last_errors = errors
                system_prompt, user_prompt = self._repair_prompt(paper_id=paper_id,
                                                                abstract=abstract,
                                                                bad_output=response,
                                                                errors=errors,
                                                                attempt=attempt)
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
            if (not did_token_retry) and self.looks_truncated(explanation_text):
                did_token_retry = True
                response = self.teacher_model.generate_response_chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=int(self.my_config.max_new_tokens_retry),  # e.g. 500 -> 750
                    do_sample=self.my_config.do_sample,
                    temperature=self.my_config.temperature,
                    top_p=self.my_config.top_p,
                    repetition_penalty=self.my_config.repetition_penalty,
                )
                gen_calls += 1
                last_response = response

                is_valid, errors, obj = self._validate_response(response, paper_id, abstract)

                # If the retry didn't help, fall back to normal repair flow
                if (not is_valid) or (not isinstance(obj, dict)):
                    attempt += 1
                    last_errors = errors if errors else ["Truncation retry produced invalid JSON."]
                    system_prompt, user_prompt = self._repair_prompt(
                        paper_id=paper_id,
                        abstract=abstract,
                        bad_output=response,
                        errors=last_errors,
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
            obj["gen_calls"] = gen_calls
            obj["did_token_retry"] = did_token_retry
            return obj


        failure_obj = {
            "id": paper_id,
            "explanation": "",
            "accepted": False,
            "attempts": attempt,
            "validation_errors": last_errors,
            "last_response": last_response,
            "gen_calls": gen_calls,
            "did_token_retry": did_token_retry,
        }
        
        if self.raise_on_fail:
            raise ValueError(
                f"Failed to generate valid response after {self.my_config.max_attempts} attempts.\n"
                f"Last response: {last_response}\n"
                f"Errors: {last_errors}"
            )
        else:
            return failure_obj
        
    def generate_batch(self, rows: List[Any]) -> List[Any]:
        system_prompt = []
        user_prompt = []
        meta_data = []
        for (paper_id, abstract) in rows:
            sp, up = self._base_prompt(paper_id, abstract)
            system_prompt.append(sp)
            user_prompt.append(up)
            meta_data.append((paper_id, abstract))
        
        raw_outputs = self.teacher_model.generate_response_chat_batch(system_prompts=system_prompt,
                                                                user_prompts=user_prompt,
                                                                max_new_tokens=self.my_config.max_new_tokens,
                                                                do_sample=self.my_config.do_sample,
                                                                temperature=self.my_config.temperature,
                                                                top_p=self.my_config.top_p,
                                                                repetition_penalty=self.my_config.repetition_penalty)
        
        results = []
        

        for (paper_id, abstract), raw in zip(meta_data, raw_outputs):
            is_valid, errors, obj = self._validate_response(raw, paper_id, abstract)

            if not (is_valid and isinstance(obj, dict)):
                # invalid JSON or schema => fallback
                results.append(self.generate_item(paper_id=paper_id, abstract=abstract))
                continue

            explanation_text = (obj.get("explanation") or "").strip()

            if self.looks_truncated(explanation_text):
                results.append(self.generate_item(paper_id=paper_id, abstract=abstract))
                continue

            ok_ctx, score_ctx, problems_ctx, obj_ctx = self._run_context_check(
                paper_id=paper_id,
                abstract=abstract,
                explanation_text=explanation_text,
            )

            if not ok_ctx:
                results.append(self.generate_item(paper_id=paper_id, abstract=abstract))
                continue

            obj["accepted"] = True
            obj["attempts"] = 1
            obj["validation_errors"] = []
            obj["context_score"] = score_ctx
            obj["context_problems"] = problems_ctx
            obj["context_judge"] = obj_ctx
            obj["gen_calls"] = 1
            obj["did_token_retry"] = False
            results.append(obj)

        return results
        
class Teacher_Data_Pipeline:
    def __init__(self, database_path, output_path, prompt_path,
                 teacher_model, my_config,
                raise_on_fail = False,
                log_skips=False,        
                print_every=1,
                max_total=None):
        
        self.database_path = database_path
        self.output_path = output_path
        self.prompt_path = prompt_path
        self.my_config = my_config     
        self.raise_on_fail = raise_on_fail
        self.log_skips = log_skips
        self.print_every = print_every
        self.max_total = max_total
        self.data_batch_size = self.my_config.data_batch_size
        self.llm_batch_size = self.my_config.llm_batch_size
        self.llm_batch = self.my_config.llm_batch
        self.teacher_model = teacher_model
        
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
        self.validator = Validation_Regeneration(
                    teacher_model=self.teacher_model,
                    my_config=self.my_config,
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
                    "teacher_model": getattr(self.teacher_model, "model_id", None),
                    "max_attempts": self.my_config.max_attempts,
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
                        LIMIT ?""", (self.data_batch_size,))
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
                (self._last_id, self.data_batch_size),)
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
        assert self.teacher_model is not None and self.validator is not None

        processed_count = 0
        success_count = 0
        fail_count = 0
        skip_count = 0
        conn = sqlite3.connect(self.database_path)
        
        try:
            self._total_eligible = self._count_total_eligible(conn)
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
            stop_now = False
            while True:
                rows = self.fetch_batch(conn)
                if not rows:
                    break
                
                if self.llm_batch:
                    # LLM batch processing
                    batching_rows = []
                    for paper_id, title_clean, abstract_clean in rows:
                        if not abstract_clean or not str(abstract_clean).strip():
                            fail_payload = {
                                "id": paper_id,
                                "title": title_clean,
                                "input": abstract_clean,
                                "error": "Empty abstract_clean",
                            }
                            self._append_jsonl(self.fail_path, fail_payload)
                            self._log_event(status="fail", paper_id=paper_id, reason="empty_abstract")

                            self.mark_processed(paper_id)
                            fail_count += 1
                            processed_count += 1

                            if self.max_total is not None and processed_count >= self.max_total:
                                stop_now = True
                                break
                            continue
                        # already processed => skip
                        if paper_id in self._processed:
                            self._last_id = paper_id
                            self.cursor_path.write_text(paper_id, encoding="utf-8")
                            skip_count += 1
                            if self.log_skips:
                                self._log_event(status="skip", paper_id=paper_id)
                            continue

                        batching_rows.append((paper_id, title_clean, abstract_clean))
                    if stop_now:
                        break
                    
                    for i in range(0, len(batching_rows), self.llm_batch_size):
                        if stop_now:
                            break
                        chunk = batching_rows[i:i + self.llm_batch_size]
                        batch_inputs = [(paper_id, abstract_clean) for (paper_id, title_clean, abstract_clean) in chunk]
                        batch_outputs = self.validator.generate_batch(batch_inputs)
                        
                        for (pid, title, abs_), out in zip(chunk, batch_outputs):
                            payload = {
                                "id": pid,
                                "title": title,
                                "input": abs_,
                                "output": out,
                                "meta": {
                                    "teacher_model": getattr(self.teacher_model, "model_id", None),
                                    "max_attempts": self.my_config.max_attempts,
                                    "llm_batch_size": self.llm_batch_size,
                                },
                            }

                            if isinstance(out, dict) and out.get("accepted"):
                                self._append_jsonl(self.train_path, payload)
                                success_count += 1
                                self._log_event(status="ok", paper_id=pid)
                            else:
                                self._append_jsonl(self.fail_path, payload)
                                fail_count += 1
                                self._log_event(status="fail", paper_id=pid, error=str(out)[:200])

                            self.mark_processed(pid)
                            processed_count += 1

                            if self.max_total is not None and processed_count >= self.max_total:
                                stop_now = True
                                break

                        # progress summary update
                        if processed_count % max(1, self.print_every) == 0:
                            pct = (processed_count / self._total_eligible) * 100.0 if self._total_eligible else None
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
                            if pct is not None:
                                print(
                                    f"[Teacher_Data_Pipeline] processed={processed_count} "
                                    f"success={success_count} fail={fail_count} skipped={skip_count} "
                                    f"percent={pct:.2f}% last_id={self._last_id}"
                                )
                            else:
                                print(
                                    f"[Teacher_Data_Pipeline] processed={processed_count} "
                                    f"success={success_count} fail={fail_count} skipped={skip_count} "
                                    f"last_id={self._last_id}"
                                )
                else:
                    for paper_id, title_clean, abstract_clean in tqdm(
                        rows, desc="Teacher pipeline", unit="paper", leave=False):
                        
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

                            if self.max_total is not None and processed_count >= self.max_total:
                                stop_now = True
                                break
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
                        if self.max_total is not None and processed_count >= self.max_total:
                            stop_now = True
                            break
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
                    if stop_now:
                        break           

            # final summary
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
            
        
