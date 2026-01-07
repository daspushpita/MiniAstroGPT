from __future__ import annotations

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
                torch_dtype=torch_dtype,
                device_map={"": "mps"},
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            
        self.model.eval()
        return self.tokenizer, self.model, self.device
    
    def generate_response(self, tokenizer, model, device,
                        prompt: str,
                        max_new_tokens: int = 512,
                        temperature : float = 0.7,
                        top_p: float = 0.9,
                        repetition_penalty: float = 1.1,
                        do_sample=True) -> str:
        """Generates responses from a given prompt

        Args:
            tokenizer (_type_): The tokenizer instance
            model (_type_): The model instance
            prompt (str): The input prompt string
        """
        # 1) tokenize
        inputs = tokenizer(prompt, return_tensors="pt")

        # 2) move to device (MPS or CPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # keep this length so we can slice out only the new tokens later
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            )
        # 3. Decode the output tokens to text
        output_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return output_text.strip()
    