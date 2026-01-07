from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from Project3.src import model

class StudentGenerator:
    def __init__(self, prompts, device: str | None = None, model_name="gpt2"):
        """
        prompts: PromptLibrary instance
        Loads tokenizer + model and keeps them.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.prompts = prompts

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        

    def generate(self, abstract: str, *, max_new_tokens: int = 350, temperature: float = 0.8,
                 top_p: float = 0.95, repetition_penalty: float = 1.1) -> str:
        """
        1) prompt = prompts.build_student_prompt(abstract)
        2) model.generate(...)
        3) decode
        4) strip everything before EXPLANATION:
        """
        prompt = self.prompts.build_student_prompt(abstract)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = self._strip_to_explanation(decoded_text)
        return explanation
        
        
    def _strip_to_explanation(self, decoded_text: str) -> str:
        """Internal helper; safest place to keep your parsing logic."""
        marker = self.prompts.EXPLANATION_MARKER
        if marker in decoded_text:
            return decoded_text.split(marker, 1)[1].strip()
        else:
            return decoded_text.strip()