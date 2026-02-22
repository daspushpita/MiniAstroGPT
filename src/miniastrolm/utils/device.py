from __future__ import annotations

import torch


def resolve_device(device: str | None) -> str:
    cfg_device = (device or "auto").lower()
    if cfg_device == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        resolved = cfg_device
        if resolved.startswith("cuda") and not torch.cuda.is_available():
            resolved = "cpu"
        elif resolved == "mps" and not (
            torch.backends.mps.is_available() and torch.backends.mps.is_built()
        ):
            resolved = "cpu"
    return resolved
