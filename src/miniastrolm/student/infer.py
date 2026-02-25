from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import time
import argparse, yaml
import argparse, yaml
from miniastrolm.utils.device import resolve_device
from miniastrolm.student.prompting import build_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftConfig, PeftModel

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
    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device,
        gen_cfg: Optional[GenerationConfig] = None,
        debug: bool = False,
        prompt_max_tokens: Optional[int] = None,
        prompt_tail_tokens: Optional[int] = None,
        base_model: Optional[str] = None,
        merge_lora: bool = False,
    ):
    
        self.model_dir = Path(model_dir)
        self.gen_cfg = gen_cfg or GenerationConfig(
            max_new_tokens=768,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=False,
        )
        self.device = device
        self.debug = debug
        self.prompt_max_tokens = prompt_max_tokens
        self.prompt_tail_tokens = prompt_tail_tokens
        self.base_model = base_model
        self.merge_lora = merge_lora
        self.model = None
        self.tokenizer = None

    def _is_adapter_checkpoint(self, model_path: Path) -> bool:
        return (model_path / "adapter_config.json").exists()

    def _resolve_base_model_name(self, model_path: Path) -> str:
        if self.base_model:
            return self.base_model
        cfg = PeftConfig.from_pretrained(str(model_path))
        base_model_name = getattr(cfg, "base_model_name_or_path", None)
        if not base_model_name:
            raise ValueError(
                "LoRA adapter detected but base model could not be resolved. "
                "Pass --base_model (e.g., gpt2-medium)."
            )
        return str(base_model_name)

    def _load_tokenizer(self, model_path: Path, fallback_name: Optional[str] = None):
        tokenizer_files_exist = (model_path / "tokenizer_config.json").exists()
        if tokenizer_files_exist:
            return AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
        if fallback_name is None:
            raise FileNotFoundError(
                f"Tokenizer files not found in {model_path}. "
                "Save tokenizer with the adapter checkpoint or pass a loadable base model."
            )
        return AutoTokenizer.from_pretrained(str(fallback_name), use_fast=True)

    def _load_model(self, model_path: Path):
        if self._is_adapter_checkpoint(model_path):
            base_model_name = self._resolve_base_model_name(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base_model, str(model_path), local_files_only=True)
            if self.merge_lora and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
            return model, base_model_name, True

        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
        return model, getattr(model.config, "_name_or_path", str(model_path)), False

    def setup(self):
        """
        Loads model/tokenizer and prepares device.
        """
        self.device = torch.device(resolve_device(self.device))
        model_path = self.model_dir.expanduser().resolve()
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        has_full_model = (model_path / "config.json").exists()
        has_adapter = (model_path / "adapter_config.json").exists()
        has_tokenizer = (model_path / "tokenizer_config.json").exists()
        if not (has_full_model or has_adapter):
            raise FileNotFoundError(
                f"Model directory {model_path} is missing both config.json and adapter_config.json."
            )
        self.model_dir = model_path
        base_model_name = self.base_model if self.base_model else None
        if has_adapter and base_model_name is None:
            base_model_name = self._resolve_base_model_name(self.model_dir)
        self.tokenizer = self._load_tokenizer(self.model_dir, fallback_name=base_model_name)
        model, resolved_name, is_adapter = self._load_model(self.model_dir)
        model.config.use_cache = True
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model = model.to(self.device)
        self.model.eval()
        
        print("Loading model from:", self.model_dir.resolve())
        print("Files:", [p.name for p in self.model_dir.iterdir()])
        print("Model name_or_path:", resolved_name)
        print("Checkpoint type:", "LoRA adapter" if is_adapter else "full model")
        if self.debug:
            print("Resolved device:", self.device)
            print("use_cache:", self.model.config.use_cache)
            print("prompt_max_tokens:", self.prompt_max_tokens)
            print("prompt_tail_tokens:", self.prompt_tail_tokens)
            print("has_tokenizer_files:", has_tokenizer)
            print("merge_lora:", self.merge_lora)

        return self.model, self.tokenizer

    def pick_device(self) -> torch.device:
        """
        Inference device selection. Keep simple.
        """
        if isinstance(self.device, torch.device):
            return self.device
        return torch.device(resolve_device(self.device))

    def _synchronize_device(self) -> None:
        if not isinstance(self.device, torch.device):
            return
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def _effective_max_new_tokens(self, prompt_len: int) -> int:
        if self.model is None:
            raise ValueError("Model is None, Initialize it first")
        max_positions = (
            getattr(self.model.config, "n_positions", None)
            or getattr(self.model.config, "max_position_embeddings", None)
        )
        requested = self.gen_cfg.max_new_tokens
        if max_positions is None:
            return requested
        if prompt_len >= max_positions:
            raise ValueError(
                f"Prompt too long ({prompt_len} tokens) for model context ({max_positions})."
            )
        available = max_positions - prompt_len
        effective = min(requested, available)
        if effective < requested:
            print(
                f"Warning: truncating max_new_tokens from {requested} to {effective} "
                f"to fit model context ({max_positions})."
            )
        return effective

    def _truncate_prompt_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if self.prompt_max_tokens is not None:
            if self.prompt_max_tokens <= 0:
                raise ValueError("prompt_max_tokens must be > 0 when provided.")
            if input_ids.shape[1] > self.prompt_max_tokens:
                if self.debug:
                    print(
                        f"Truncating prompt tokens from {input_ids.shape[1]} "
                        f"to {self.prompt_max_tokens} (tail)."
                    )
                input_ids = input_ids[:, -self.prompt_max_tokens :]
                attention_mask = attention_mask[:, -self.prompt_max_tokens :]

        if self.prompt_tail_tokens is not None:
            if self.prompt_tail_tokens <= 0:
                raise ValueError("prompt_tail_tokens must be > 0 when provided.")
            if input_ids.shape[1] > self.prompt_tail_tokens:
                if self.debug:
                    print(
                        f"Keeping only prompt tail tokens: {self.prompt_tail_tokens} "
                        f"(from {input_ids.shape[1]})."
                    )
                input_ids = input_ids[:, -self.prompt_tail_tokens :]
                attention_mask = attention_mask[:, -self.prompt_tail_tokens :]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate_one(self, abstract: str, paper_id: Optional[str] = None) -> str:
        """
        Returns only the generated explanation text.
        """
        if abstract is None or not abstract.strip():
            raise ValueError("Abstract cannot be empty. Provide --abstract.")
        prompt = build_prompt(self.tokenizer, paper_id or "", abstract)
        # 1) tokenize
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None, Initialize it first")
        if self.model is None:
            raise ValueError("Model is None, Initialize it first")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = self._truncate_prompt_inputs(inputs)
        # 2) move to device (MPS or CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        effective_max_new_tokens = self._effective_max_new_tokens(prompt_len)
        if self.debug:
            print("Prompt tokens:", prompt_len)
            print("Requested max_new_tokens:", self.gen_cfg.max_new_tokens)
            print("Effective max_new_tokens:", effective_max_new_tokens)

        self._synchronize_device()
        started = time.perf_counter()
        gen_kwargs = {
            **inputs,
            "max_new_tokens": effective_max_new_tokens,
            "do_sample": self.gen_cfg.do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.gen_cfg.do_sample:
            gen_kwargs["temperature"] = self.gen_cfg.temperature
            gen_kwargs["top_p"] = self.gen_cfg.top_p
        if self.gen_cfg.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.gen_cfg.repetition_penalty
        outputs = self.model.generate(**gen_kwargs)
        self._synchronize_device()
        elapsed_s = time.perf_counter() - started

        generated_ids = outputs[0][prompt_len:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if self.debug:
            print("Generated tokens:", generated_ids.shape[0])
            print(f"Generation time: {elapsed_s:.2f}s")
            if generated_ids.shape[0] == 0:
                print("Warning: model generated 0 new tokens (likely immediate EOS).")
            elif not output_text.strip():
                print("Warning: generated tokens decode to empty text after stripping.")
        return output_text.strip()

    @torch.no_grad()
    def generate_many(self, abstracts: List[str]) -> List[str]:
        # TODO: loop or batch
        pass

def load_config(path: str| Path) -> GenerationConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping.")
    cfg = raw.get("generation", raw)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping for generation settings.")

    return GenerationConfig(
        max_new_tokens=int(cfg.get("max_new_tokens", 768)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
        do_sample=bool(cfg.get("do_sample", False)),
    )

def parse_args():
    p = argparse.ArgumentParser(description="Generate explanation from the student model")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--model_dir", type=Path, required=True, help="path to the student model files")
    p.add_argument("--device", type=str, default=None, help="Override device: auto/cpu/mps/cuda")
    p.add_argument("--abstract", type=str, default=None, help="Give the abstract")
    p.add_argument("--paper_id", type=str, default="", help="Paper ID to include in the prompt JSON schema")
    p.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens from config")
    p.add_argument("--debug", action="store_true", help="Print generation diagnostics")
    p.add_argument("--print_repr", action="store_true", help="Print repr(output) to debug blank outputs")
    p.add_argument(
        "--prompt_max_tokens",
        type=int,
        default=None,
        help="Tail-truncate prompt to this many tokens (set to train max_length, e.g., 768).",
    )
    p.add_argument(
        "--prompt_tail_tokens",
        type=int,
        default=None,
        help="Optional extra tail cap to mirror training prefix tail (e.g., collator min_prefix_tokens=128).",
    )
    p.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Optional base model name/path when loading a LoRA adapter checkpoint (e.g., gpt2-medium).",
    )
    p.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA adapter into the base model for inference after loading.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    generation_config = load_config(args.config)
    if args.max_new_tokens is not None:
        generation_config.max_new_tokens = int(args.max_new_tokens)
    infer = StudentInferencer(
        model_dir=args.model_dir,
        device=args.device,
        gen_cfg=generation_config,
        debug=args.debug,
        prompt_max_tokens=args.prompt_max_tokens,
        prompt_tail_tokens=args.prompt_tail_tokens,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
    )
    infer.setup()
    output = infer.generate_one(args.abstract, paper_id=args.paper_id)
    if args.print_repr:
        print(repr(output))
    else:
        print(output)


if __name__ == "__main__":
    main()
