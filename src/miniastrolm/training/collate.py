import argparse
import random
import torch
import os
from typing import List, Dict


class CausalLMCollator:
    """
    Collator for causal language modeling.

    Turns:
        [{"id": ..., "text": ...}, ...]

    Into:
        {
          "input_ids": Tensor,
          "attention_mask": Tensor,
          "labels": Tensor
        }
    """
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, str]]):
        texts = [item["text"] for item in batch]
        my_encodings = self.tokenizer(texts,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt")
        
        labels = my_encodings["input_ids"].clone()
        labels[my_encodings["attention_mask"] == 0] = -100
        my_encodings["labels"] = labels
        
        my_encodings["ids"] = [item["id"] for item in batch]
        return my_encodings