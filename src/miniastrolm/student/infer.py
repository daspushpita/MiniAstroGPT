from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import torch

from .device import resolve_device

from transformers import AutoTokenizer, AutoModelForCausalLM

# 1.	Load a saved student checkpoint (model + tokenizer)
# 2.	Format inference prompts to match training (Abstract:\n...\n\nExplanation:\n)
# 3.	Generate the continuation (the explanation) and return it

@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    do_sample: bool

class StudentInferencer:
    def __init__(self, model_dir: str | Path, device: str | torch.device, 
                gen_cfg: Optional[GenerationConfig] = None):
    
        self.model_dir = Path(model_dir)
        self.gen_cfg = gen_cfg or GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
        self.device = device
        self.model = None
        self.tokenizer = None


    def setup(self):
        """
        Loads model/tokenizer and prepares device.
        """
        self.device = torch.device(resolve_device(self.device))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model = model.to(self.device)
        self.model.eval()
        
        return self.model, self.tokenizer

    def format_prompt(self, abstract: str) -> str:
        """
        Must match training format exactly.
        """

        return f"Abstract:\n{abstract}\n\nExplanation:\n"


    def pick_device(self) -> torch.device:
        """
        Inference device selection. Keep simple.
        """
        if isinstance(self.device, torch.device):
            return self.device
        return torch.device(resolve_device(self.device))

    @torch.no_grad()
    def generate_one(self, abstract: str) -> str:
        """
        Returns only the generated explanation text.
        """
        prompt = self.format_prompt(abstract)
        # 1) tokenize
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None, Initialize it first")
        if self.model is None:
            raise ValueError("Model is None, Initialize it first")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # 2) move to device (MPS or CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=self.gen_cfg.max_new_tokens,
                                temperature=self.gen_cfg.temperature,
                                top_p=self.gen_cfg.top_p,
                                repetition_penalty=self.gen_cfg.repetition_penalty,
                                do_sample=self.gen_cfg.do_sample
                            )
        output_text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return output_text.strip()

    @torch.no_grad()
    def generate_many(self, abstracts: List[str]) -> List[str]:
        # TODO: loop or batch
        pass


def main() -> None:
    # TODO:
    # - parse args: --model_dir, --device, --abstract or --input_jsonl
    # - infer = StudentInferencer(...)
    # - infer.setup()
    # - print outputs
    pass


if __name__ == "__main__":
    main()