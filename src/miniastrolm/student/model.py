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
    
def freeze_gpt2_bottom(model, n_freeze_blocks: int, freeze_embeddings: bool = True):
    """
    Freeze GPT-2 embeddings + first n transformer blocks.
    Keeps later blocks, ln_f, and lm_head trainable by default.
    """
    # Freeze embeddings
    if freeze_embeddings:
        model.transformer.wte.requires_grad_(False)
        model.transformer.wpe.requires_grad_(False)
    
    # Freeze first N blocks
    n_freeze_blocks = max(0, min(n_freeze_blocks, len(model.transformer.h)))
    for i in range(n_freeze_blocks):
        model.transformer.h[i].requires_grad_(False)
    
    # Ensure top stays trainable (optional explicit)
    for i in range(n_freeze_blocks, len(model.transformer.h)):
        model.transformer.h[i].requires_grad_(True)

    # Keep final norm + head trainable
    model.transformer.ln_f.requires_grad_(True)
    if hasattr(model, "lm_head"):
        model.lm_head.requires_grad_(True)

    return model