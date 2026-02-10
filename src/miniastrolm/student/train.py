from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional
import argparse
import numpy as np
import random
import torch

from .device import resolve_device
from torch.utils.data import DataLoader
import os, yaml
from tqdm import tqdm
from pathlib import Path
from miniastrolm.student.data import JsonlStudentDataset
from miniastrolm.student.model import load_student
from miniastrolm.training.collate import CausalLMCollator
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelConfig:
    model_name: str
    max_length: int
    
@dataclass
class DataConfig:
    train_path: str
    max_samples: int

@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    max_steps: int
    weight_decay: float
    seed: int
    device: str

@dataclass
class OutputConfig:
    output_dir: str

@dataclass
class MainConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig
    
    
class TrainRunner:
    """
    Orchestrates student training.
    """
    def __init__(self, cfg: MainConfig, debug : bool = False):
        self.config = cfg
        self.device = self.config.training.device
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.collator = None
        self.optimizer = None
        self.debug = debug
        
    # ---- setup steps ----
    def setup_seed(self, seed:int=42):
        """Sets the seed for generating random numbers in PyTorch, NumPy, and Python."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set as {seed}")


    def setup_device(self):
        self.device = resolve_device(self.config.training.device)
        print(f"Using device: {self.device}")
        return self.device

    def setup_model(self):
        self.model, self.tokenizer = load_student(self.config.model.model_name, self.config.model.max_length)
        self.model = self.model.to(self.device)
        print(f"Model on device: {next(self.model.parameters()).device}")
        self.model.train()
        return self.model

    def setup_data(self):
        
        self.dataset = JsonlStudentDataset(path = self.config.data.train_path,
                input_key = "abstract",
                output_key = "target_explanation",
                id_key = "id",
                strip_whitespace = True,
                max_samples = self.config.data.max_samples)
        if self.debug:
            print("Number of samples:", len(self.dataset))
            print("First sample preview:")
            print(self.dataset[0]["text"][:300])
            
        self.collator = CausalLMCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.model.max_length,
        )
        num_workers = 0 #min(4, os.cpu_count() or 0)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory = (str(self.device).startswith("cuda")),
            persistent_workers=num_workers > 0,
            collate_fn=self.collator,
        )
        if self.debug:
            batch = next(iter(self.dataloader))
            print("Batch keys:", batch.keys())
        return self.dataloader

    def setup_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr = self.config.training.lr,
                                        weight_decay=self.config.training.weight_decay,
                                        eps=1e-8)
        
        return self.optimizer

    # ---- core loop ----

    def train_loop(self):
        device = self.device
        model = self.model
        optimizer = self.optimizer
        dataloader = self.dataloader
        if model is None or optimizer is None or dataloader is None:
            raise RuntimeError("Model, optimizer, and dataloader must be initialized before training.")
        max_steps = self.config.training.max_steps
        log_every = max(1, min(50, max_steps // 10 or 1))
        
        step = 0
        running_loss = 0.0
        
        if self.debug:
            batch = next(iter(dataloader))
            ids = batch.pop("ids", None)   # keep for logging if you want
            batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, "to")}
            # if you still want ids later, keep it separately
            # batch = {k: v.to(self.device) for k, v in batch.items()}
            out = model(**batch)
            print("Loss:", out.loss)
            
        with tqdm(total=max_steps, unit="step") as pbar:
            while step < max_steps:
                for batch in dataloader:
                    step += 1
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    running_loss += loss.item()
                    if step == 1 or step % log_every == 0 or step == max_steps:
                        avg_loss = running_loss / max(
                            1, log_every if step % log_every == 0 else 1
                        )
                        pbar.set_postfix(loss=f"{avg_loss:.4f}")
                        running_loss = 0.0

                    pbar.update(1)
                    if step >= max_steps:
                        break
                    if (step % 10 == 0):
                        print(f"step={step} loss={loss.item():.4f}")
                    
                    if self.debug and step == 1:
                        print("One backward step succeeded")
                    

    # ---- lifecycle ----

    def run(self) -> None:
        self.setup_seed(self.config.training.seed)
        self.setup_device()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.train_loop()
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be initialized before saving.")
        self.model.save_pretrained(self.config.output.output_dir)
        self.tokenizer.save_pretrained(self.config.output.output_dir)


# ---------------- Config loading ----------------
def load_config(path: str | Path) -> MainConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping.")

    def require_section(name: str) -> dict:
        sec = raw.get(name)
        if not isinstance(sec, dict):
            raise ValueError(f"Missing or invalid section '{name}'")
        return sec

    model = ModelConfig(**require_section("model"))
    data = DataConfig(**require_section("data"))
    training = TrainingConfig(**require_section("training"))
    output = OutputConfig(**require_section("output"))

    return MainConfig(model=model, data=data, training=training, output=output)


# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Train AstroGPT student model")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")

    # optional overrides (only include what you actually want)
    p.add_argument("--device", type=str, default=None, help="Override device: auto/cpu/mps/cuda")
    p.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.training.device = args.device
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    runner = TrainRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
        
        
