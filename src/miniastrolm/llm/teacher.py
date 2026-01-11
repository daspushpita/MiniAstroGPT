from __future__ import annotations

import json
from pathlib import Path
from urllib import response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import Dict, List, Tuple, Any

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
                        max_new_tokens: int = 512,
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
    
class Validation_Regeneration:
    def __init__(self, teacher_model: Llama_Teacher, max_attempts: int = 3, 
                prompt_path : Path | None = None):
        self.teacher_model = teacher_model
        self.max_attempts = max_attempts
        self.min_chars = 800
        self.banned_phrases = [
            "we present", "we show", "we find", "we propose", "we investigate",
            "in this paper", "this paper", "we study", "we explore",
            "we report", "we demonstrate",
        ]
        if prompt_path is None:
            raise ValueError("Base Teacher prompt path must be provided.")
        self.base_prompt_template = prompt_path.read_text(encoding="utf-8").strip()

    def _base_prompt(self, paper_id: str, abstract: str) -> str:
        return self.base_prompt_template.format(
            paper_id=paper_id,
            abstract=abstract,
        )

    def _validate_response(self, explanation: str, paper_id: str, abstract: str) ->  Tuple[bool, List[str], Dict[str, Any] | None]:
        errors: List[str] = []
        text = (explanation or "").strip()
        
        try:
            obj = json.loads(text)
        except Exception as e:
            snippet = text[:200].replace("\n", "\\n")
            return False, [f"Invalid JSON: {e}. Snippet: {snippet!r}"], None
        
        if not isinstance(obj, dict):
            errors.append("Response is not a JSON object")
            return False, errors, None

        if "id" not in obj or "explanation" not in obj:
            return False, ['JSON must contain keys: "id" and "explanation".'], None
    
        if str(obj["id"]) != str(paper_id):
            return False, [f"Paper ID mismatch: expected {paper_id}, got {obj['id']}"], None

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

        return (len(errors) == 0), errors, (obj if len(errors) == 0 else None)
    
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
        errors_bullets = "\n".join(f"- {e}" for e in errors)
        return f"""Your previous explanation FAILED validation.

PAPER ID: "{paper_id}"

VALIDATION ERRORS:
{errors_bullets}

HARD REQUIREMENTS:
- Output ONLY valid JSON. No markdown. No commentary. No extra text.
- Top-level output MUST be a single JSON object (not an array).
- The JSON object MUST have exactly these keys: "id", "explanation".
- "id" MUST equal: {paper_id}
- "explanation" MUST be at least {self.min_chars} characters.
- Must NOT use academic-reporting phrases (e.g., "we present", "in this paper", etc.).
- Must be rewritten from scratch and not copy phrases from the abstract.

ABSTRACT:
{abstract}

YOUR INVALID OUTPUT (for reference only — do NOT reuse wording):
{bad_output}

Now output ONLY this JSON object, starting with '{{' and ending with '}}':
{{"id":"{paper_id}","explanation":"..."}}
"""


    def generate_item(self, paper_id: str, abstract: str) -> Dict[str, Any]:
        attempt = 0
        last_response = ""
        last_errors: List[str] = []
        prompt = self._base_prompt(paper_id, abstract)
        
        while attempt < self.max_attempts:
            response = self.teacher_model.generate_response(prompt=prompt)
            last_response = response
            is_valid, errors, obj = self._validate_response(response, paper_id, abstract)
            if is_valid:
                return obj
            last_errors = errors
            attempt += 1
            prompt = self._repair_prompt(
                paper_id=paper_id,
                abstract=abstract,
                bad_output=response,
                errors=errors,
                attempt=attempt,
            )
        raise ValueError(
            f"Failed to generate valid response after {self.max_attempts} attempts.\n"
            f"Last response: {last_response}\n"
            f"Errors: {last_errors}"
        )
        