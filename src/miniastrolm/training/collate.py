import random
from typing import List, Dict, Any

import torch


class CausalLMCollator:
    """
    Collator for causal LM on prompt->explanation style data.

    Expected each item in batch has:
        {"id": ..., "text": ...}

    Where text contains a split marker:
        "...Explanation:\n" + <target explanation text>

    We supervise ONLY the target explanation tokens (labels=-100 for the prefix).

    Truncation policy (important):
      - Prefer keeping target (answer) tokens.
      - Always reserve at least `min_prefix_tokens` tokens for the *tail* of the prefix
        so the model sees the "Explanation:\n" boundary + some abstract context.
      - If prefix must be truncated, keep the *end* of the prefix (tail), not the start.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        min_prefix_tokens: int = 128,
        debug: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.min_prefix_tokens = int(min_prefix_tokens)
        self.debug = bool(debug)

        # GPT-2 often has no PAD token. Use EOS as PAD for batching.
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is None:
                raise ValueError("Tokenizer has no pad_token_id and no eos_token to use as pad.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = int(self.tokenizer.pad_token_id)

        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")
        if self.min_prefix_tokens < 0:
            raise ValueError("min_prefix_tokens must be >= 0")
        if self.min_prefix_tokens >= self.max_length:
            raise ValueError("min_prefix_tokens must be < max_length so there is space for target tokens.")

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any] | None:
        marker = "### Output:\n"

        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        kept_ids: List[str] = []
        bad_sample = 0

        for item in batch:
            text = item["text"]
            ex_id = item.get("id", "")

            j = text.find(marker)
            if j == -1:
                bad_sample += 1
                continue

            prefix = text[: j + len(marker)]      # includes marker
            target = text[j + len(marker):]       # target explanation

            prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

            if len(target_ids) == 0:
                bad_sample += 1
                continue

            # ---- Budget policy: KEEP TARGET, RESERVE PREFIX TAIL ----
            max_target = self.max_length - self.min_prefix_tokens
            if max_target <= 0:
                bad_sample += 1
                continue

            # 1) Keep as much target as possible (up to max_target)
            if len(target_ids) > max_target:
                target_ids = target_ids[:max_target]
                if self.tokenizer.eos_token_id is not None:
                    target_ids[-1] = self.tokenizer.eos_token_id  # preserve stop signal
                if self.tokenizer.eos_token_id is not None:
                    target_ids[-1] = self.tokenizer.eos_token_id  # ensure EOS if truncated

            # 2) Fit prefix into remaining space by keeping the TAIL
            max_prefix = self.max_length - len(target_ids)
            if max_prefix < 0:
                prefix_ids = []
            elif len(prefix_ids) > max_prefix:
                prefix_ids = prefix_ids[-max_prefix:]

            ids = prefix_ids + target_ids
            labels = [-100] * len(prefix_ids) + target_ids[:]  # supervise only target

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            kept_ids.append(ex_id)

        if len(input_ids_list) == 0:
            return None

        # ---- Pad ----
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        lengths = torch.tensor([t.size(0) for t in input_ids_list], dtype=torch.long)
        attention_mask = (
            torch.arange(input_ids.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()

        enc = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ids": kept_ids,
        }

        if self.debug:
            with torch.no_grad():
                supervised = (labels != -100).sum(dim=1)
                tok_counts = attention_mask.sum(dim=1).clamp_min(1)
                ratio = (supervised.float() / tok_counts.float()).mean().item()
                print(
                    f"[COLLATE DEBUG] kept={len(kept_ids)}/{len(batch)} "
                    f"bad={bad_sample} "
                    f"avg_tokens={tok_counts.float().mean().item():.1f} "
                    f"avg_supervised={supervised.float().mean().item():.1f} "
                    f"ratio={ratio:.2f}"
                )

        return enc
