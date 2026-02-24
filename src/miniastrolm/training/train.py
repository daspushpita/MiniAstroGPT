from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional
import argparse
import numpy as np
import random
import torch
import gc
import csv

from miniastrolm.utils.device import resolve_device
from torch.utils.data import DataLoader
import os, yaml
from tqdm import tqdm
from pathlib import Path
from miniastrolm.student.data import JsonlStudentDataset
from miniastrolm.student.model import load_student, freeze_gpt2_bottom, apply_peft, setup_optimizer
from miniastrolm.training.collate import CausalLMCollator
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelConfig:
    model_name: str
    max_length: int
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.01
    
@dataclass
class DataConfig:
    train_path: str
    max_samples: int
    val_path: Optional[str] = None
    val_max_samples: Optional[int] = None
    test_path: Optional[str] = None

@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    max_steps: int
    weight_decay: float
    seed: int
    device: str
    freeze_embeddings : bool
    gradient_accumulation_steps: int
    n_freeze_blocks: Optional[int] = None
    use_grad_accum_loss: bool = True
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
        if self.config.model.use_lora:
            self.model = apply_peft(
                self.model,
                r=self.config.model.lora_r,
                alpha=self.config.model.lora_alpha,
                dropout=self.config.model.lora_dropout,
            )
        else:
            print("LoRA disabled (model.use_lora=false); training base model directly.")

        freeze_gpt2_bottom(self.model, n_freeze_blocks=self.config.training.n_freeze_blocks, freeze_embeddings=self.config.training.freeze_embeddings)
        self.model = self.model.to(self.device)
        self.model.config.use_cache = False
        # With LoRA + gradient checkpointing, ensure input embeddings require grads.
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if str(self.device).startswith(("cuda", "mps")):
            self.model.gradient_checkpointing_enable()

        print(f"Model on device: {next(self.model.parameters()).device}")
        if not any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("No trainable parameters found after model setup.")
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
        
        self.val_dataset = None
        if self.config.data.val_path:
            self.val_dataset = JsonlStudentDataset(path = self.config.data.val_path,
                    tokenizer=self.tokenizer,
                    input_key = "abstract",
                    output_key = "target_explanation",
                    id_key = "id",
                    strip_whitespace = True,
                    max_samples = self.config.data.val_max_samples)
        
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
        training_loader = DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory = (str(self.device).startswith("cuda")),
            persistent_workers=num_workers > 0,
            collate_fn=self.collator,
        )
        validation_loader = None
        if self.val_dataset is not None:
            validation_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory = (str(self.device).startswith("cuda")),
                persistent_workers=num_workers > 0,
                collate_fn=self.collator,
            )
        if self.debug:
            batch = self._first_valid_batch(training_loader)
            if batch is None:
                print("Warning: no valid training batch found in setup_data() debug preview.")
            else:
                print("Batch keys:", batch.keys())
        return training_loader, validation_loader

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

    def batch_loss(self, model, batch, device):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        return loss

    def _first_valid_batch(self, data_loader):
        for batch in data_loader:
            if batch is not None:
                return batch
        return None


    def evaluate(self, model, val_loader, device, eval_step):
        if val_loader is None:
            return None

        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                val_loss = self.batch_loss(model, batch, device)
                total_val_loss += float(val_loss.detach().item())
                val_batches += 1

        model.train()
        if val_batches == 0:
            print(f"Evaluation at step {eval_step}: no valid validation batches")
            return None

        avg_val_loss = total_val_loss / val_batches
        print(f"Evaluation at step {eval_step}: avg_val_loss={avg_val_loss:.4f}")
        return avg_val_loss

    def _optimizer_update(self, model, optimizer) -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def _write_train_row(self, writer, optimizer_step, micro_step, update_loss, loss_for_log) -> None:
        writer.writerow(
            [
                optimizer_step,
                micro_step,
                f"{update_loss:.6f}",
                f"{loss_for_log:.6f}",
                f"{self.optimizer.param_groups[0]['lr']:.10f}",
            ]
        )

    def _should_log(self, optimizer_step, log_every, max_steps) -> bool:
        return optimizer_step == 1 or optimizer_step % log_every == 0 or optimizer_step == max_steps

    def _maybe_free_cache(self, optimizer_step, device) -> None:
        if optimizer_step % 2 == 0:
            gc.collect()
            if str(device).startswith("mps"):
                torch.mps.empty_cache()

        

    def train_loop(self):
        device = self.device
        model = self.model
        optimizer = self.optimizer
        training_loader, validation_loader = self.setup_data()
        
        self.print_training_example_once()
        if model is None or optimizer is None or training_loader is None:
            raise RuntimeError("Model, optimizer, and dataloader must be initialized before training.")
        
        max_steps = int(self.config.training.max_steps)
        grad_accum_steps = int(self.config.training.gradient_accumulation_steps)
        if not bool(self.config.training.use_grad_accum_loss):
            grad_accum_steps = 1
        
        if max_steps <= 0:
            raise ValueError("training.max_steps must be > 0")
        if grad_accum_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be > 0")
        log_every = max(1, min(50, max_steps // 10 or 1))
        
        micro_step = 0
        optimizer_step = 0
        log_loss_sum = 0.0
        log_update_count = 0
        grad_accum_sum = 0.0
        
        if self.debug:
            batch = self._first_valid_batch(training_loader)
            if batch is None:
                print("Warning: no valid training batch found for debug forward pass.")
            else:
                batch.pop("ids", None)
                batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, "to")}
                out = model(**batch)
                print("Loss:", out.loss)
            
        with self.train_loss_path.open("w", encoding="utf-8", newline="") as loss_file:
            writer = csv.writer(loss_file)
            writer.writerow(["optimizer_step", "micro_step", "update_loss", "last_micro_loss", "lr"])
            with tqdm(total=max_steps, unit="step") as pbar:
                optimizer.zero_grad(set_to_none=True)
                while optimizer_step < max_steps:
                    seen_valid_batch = False
                    for batch in training_loader:
                        if batch is None:
                            continue
                        seen_valid_batch = True
                        micro_step += 1

                        loss = self.batch_loss(model, batch, device)
                        loss_for_log = loss.detach().item()
                        
                        if not torch.isfinite(loss):
                            labels = batch["labels"].to(device, non_blocking=True)
                            print(
                                f"Critical: non-finite loss detected at micro_step={micro_step}, "
                                f"optimizer_step={optimizer_step}. Reducing LR is advised."
                            )
                            valid = (labels != -100).sum(dim=1)
                            print("valid tokens per sample:", valid.tolist())
                            print("max prompt_lens:", int((labels == -100).sum(dim=1).max().item()))
                            return

                        loss = loss / grad_accum_steps
                        loss.backward()
                        grad_accum_sum += loss_for_log
                        
                        if micro_step % grad_accum_steps == 0:
                            self._optimizer_update(model=model, optimizer=optimizer)
                            optimizer_step += 1
                            update_loss = grad_accum_sum / grad_accum_steps
                            grad_accum_sum = 0.0
                            log_loss_sum += update_loss
                            log_update_count += 1
                            self._write_train_row(
                                writer=writer,
                                optimizer_step=optimizer_step,
                                micro_step=micro_step,
                                update_loss=update_loss,
                                loss_for_log=loss_for_log,
                            )
                            if self._should_log(optimizer_step=optimizer_step, log_every=log_every, max_steps=max_steps):
                                avg_loss = log_loss_sum / max(1, log_update_count)
                                val_loss = self.evaluate(
                                    model=model,
                                    val_loader=validation_loader,
                                    device=device,
                                    eval_step=optimizer_step,
                                )
                                if val_loss is None:
                                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                                else:
                                    pbar.set_postfix(loss=f"{avg_loss:.4f}", val=f"{val_loss:.4f}")
                                log_loss_sum = 0.0
                                log_update_count = 0
                                if str(device).startswith("mps"):
                                    gc.collect()
                                    torch.mps.empty_cache()

                            self._maybe_free_cache(optimizer_step=optimizer_step, device=device)
                            pbar.update(1)
                            
                            if optimizer_step >= max_steps:
                                break

                        del loss

                        if self.debug and micro_step == 1:
                            print("One backward step succeeded")
                    if not seen_valid_batch:
                        raise RuntimeError(
                            "DataLoader produced no valid training batches. "
                            "Check collator filtering and dataset format."
                        )
                    


    def run(self) -> None:
        self.setup_seed(self.config.training.seed)
        self.setup_device()
        self.setup_model()
        self.optimizer, self.scheduler = setup_optimizer(self.model, self.config)
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        self.train_loop()
        print(f"Saved train loss log to: {self.train_loss_path}")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be initialized before saving.")
        self.model.save_pretrained(self.config.output.output_dir)
        self.tokenizer.save_pretrained(self.config.output.output_dir)
        print(f"Model and tokenizer saved to: {self.config.output.output_dir}")


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
    p.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable LoRA adapter training.",
    )
    p.add_argument(
        "--use_grad_accum_loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable gradient accumulation loss scaling.",
    )

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
    if args.use_lora is not None:
        config.model.use_lora = args.use_lora
    if args.use_grad_accum_loss is not None:
        config.training.use_grad_accum_loss = args.use_grad_accum_loss
    runner = TrainRunner(config, debug=args.debug)
    runner.run()


if __name__ == "__main__":
    main()
        
        
