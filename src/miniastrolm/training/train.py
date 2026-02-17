from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional
import argparse
import numpy as np
import random
import torch
import gc
import csv

from miniastrolm.student.device import resolve_device
from torch.utils.data import DataLoader
import os, yaml
from tqdm import tqdm
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
from miniastrolm.student.data import JsonlStudentDataset
from miniastrolm.student.model import load_student, freeze_gpt2_bottom
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
    freeze_embeddings : bool
    n_freeze_blocks: int
    scheduler: str = "linear"
    warmup_ratio: float = 0.1

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
        self.scheduler = None
        self.debug = debug
        self.train_loss_path = Path(self.config.output.output_dir) / "train_loss.csv"
        self._printed_training_example = False
        
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
        
        freeze_gpt2_bottom(self.model, n_freeze_blocks=self.config.training.n_freeze_blocks, freeze_embeddings=self.config.training.freeze_embeddings)
        self.model = self.model.to(self.device)

        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

        print(f"Model on device: {next(self.model.parameters()).device}")
        self.model.train()
        return self.model

    def setup_data(self):
        
        self.dataset = JsonlStudentDataset(path = self.config.data.train_path,
                tokenizer=self.tokenizer,
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
            debug=self.debug,
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
        # ex = self.dataset[0]
        # print(ex.keys())
        # print("TEXT PREVIEW:\n", ex["text"])
        return self.dataloader

    def setup_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW( # Use AdamW
            trainable_params, 
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            eps=1e-8
        )
        scheduler_name = str(self.config.training.scheduler).lower()
        if scheduler_name == "linear":
            warmup_steps = int(self.config.training.warmup_ratio * self.config.training.max_steps)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.config.training.max_steps
            )
        elif scheduler_name in {"constant", "none"}:
            self.scheduler = None
        else:
            raise ValueError(
                f"Unsupported scheduler '{self.config.training.scheduler}'. "
                "Use one of: linear, constant, none."
            )
        return self.optimizer

    # ---- core loop ----
    def print_training_example_once(self) -> None:
        if self._printed_training_example:
            return
        train_path = Path(self.config.data.train_path)
        try:
            with train_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        print(f"Training example ({train_path}):")
                        print(line)
                        self._printed_training_example = True
                        return
            print(f"Training example ({train_path}): <empty file>")
        except Exception as exc:
            print(f"Warning: could not print training example from {train_path}: {exc}")
        self._printed_training_example = True

    def train_loop(self):
        device = self.device
        model = self.model
        optimizer = self.optimizer
        dataloader = self.dataloader
        self.print_training_example_once()
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
            
        with self.train_loss_path.open("w", encoding="utf-8", newline="") as loss_file:
            writer = csv.writer(loss_file)
            writer.writerow(["step", "loss", "lr"])
            with tqdm(total=max_steps, unit="step") as pbar:
                while step < max_steps:
                    for batch in dataloader:
                        if batch is None:
                            continue
                        step += 1
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        labels = batch["labels"].to(device, non_blocking=True)
                        
                        outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                        loss = outputs.loss
                        if torch.isnan(loss):
                            print(f"Critcal: NaN loss detected at step {step}. Reducing LR is advised.")
                            valid = (labels != -100).sum(dim=1)
                            print("valid tokens per sample:", valid.tolist())
                            print("max prompt_lens:", int((labels == -100).sum(dim=1).max().item()))
                            return

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Add this line
                        optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        running_loss += loss.item()
                        writer.writerow([step, f"{loss.item():.6f}", f"{self.optimizer.param_groups[0]['lr']:.10f}"])
                        if step == 1 or step % log_every == 0 or step == max_steps:
                            avg_loss = running_loss / max(
                                1, log_every if step % log_every == 0 else 1
                            )
                            pbar.set_postfix(loss=f"{avg_loss:.4f}")
                            running_loss = 0.0

                        if step % 50 == 0:
                            gc.collect()
                            torch.mps.empty_cache()
                        pbar.update(1)
                        if step >= max_steps:
                            break
                        if (step % 10 == 0):
                            print(f"step={step} loss={loss.item():.4f}")
                            
                        del outputs, loss, input_ids, attention_mask, labels

                        if self.debug and step == 1:
                            print("One backward step succeeded")
                    

    # ---- lifecycle ----

    def run(self) -> None:
        self.setup_seed(self.config.training.seed)
        self.setup_device()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        self.train_loop()
        print(f"Saved train loss log to: {self.train_loss_path}")
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
    p.add_argument("--debug", action="store_true", help="Enable training + collator debug prints")

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
    runner = TrainRunner(config, debug=args.debug)
    runner.run()


if __name__ == "__main__":
    main()
        
        
