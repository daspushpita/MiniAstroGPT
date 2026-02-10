from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_student(model_name: str,
            max_length: int):
    """
    Load student model + tokenizer.

    Responsibilities:
    - load HF tokenizer
    - load HF causal LM
    - fix pad token issues (GPT-2)
    - set max_length related config if needed

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # GPT-2 has no pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.config.use_cache = False
    return model, tokenizer
    
