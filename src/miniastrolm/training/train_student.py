from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class StudentTrainer:
    def __init__(self, model,
                 tokenizer,
                 prompt_library,
                 device: str | None = None):
        """

        Args:
            model (_type_): _description_
            tokenizer (_type_): _description_
            prompt_library (_type_): _description_
            device (str | None, optional): _description_. Defaults to None.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_library = prompt_library
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        
    def make_example(self, abstract: str, explanation: str) -> dict:
        """Returns dict with input_ids, attention_mask, labels."""
        ...

    def fit(self, train_items: list[dict], *, epochs: int, batch_size: int, lr: float):
        ...