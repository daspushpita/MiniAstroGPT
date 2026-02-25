from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
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

def apply_peft(model, r: int, alpha: int, dropout: float):
    """
    Apply LoRA to the model.

    By default, applies to all linear layers in the transformer blocks.
    Can be customized to target specific layers or modules if desired.
    """
    peft_config = LoraConfig(r=r,  # Rank
                    lora_alpha=alpha,  # Scaling factor
                    target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 specific attention layer
                    lora_dropout=dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def setup_optimizer(model, config):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, 
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        eps=1e-8)
    
    scheduler_name = str(config.training.scheduler).lower()
    if scheduler_name == "linear":
        # max_steps is interpreted as optimizer-update steps (not micro-batches).
        warmup_steps = int(config.training.warmup_ratio * config.training.max_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.training.max_steps
        )
    elif scheduler_name in {"constant", "none"}:
        scheduler = None
    else:
        raise ValueError(
            f"Unsupported scheduler '{config.training.scheduler}'. "
            "Use one of: linear, constant, none."
        )
    return optimizer, scheduler
    
def freeze_gpt2_bottom(model, n_freeze_blocks: Optional[int] = None, freeze_embeddings: bool = True):
    """
    Optionally freeze GPT-2 embeddings and the first n transformer blocks.
    If n_freeze_blocks is None, only embedding freezing is applied.
    """
    # Freeze embeddings
    if freeze_embeddings:
        model.transformer.wte.requires_grad_(False)
        model.transformer.wpe.requires_grad_(False)
    
    # Optional block freezing: if not set, keep current trainability as-is.
    if n_freeze_blocks is None:
        return model

    # Freeze first N blocks
    n_freeze_blocks = max(0, min(n_freeze_blocks, len(model.transformer.h)))
    for i in range(n_freeze_blocks):
        model.transformer.h[i].requires_grad_(False)

    return model
