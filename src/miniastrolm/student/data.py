
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable
import json
from torch.utils.data import Dataset

# ---------- Data container ----------
@dataclass(frozen=True)
class Sample:
    id: str
    abstract: str
    target_explanation: str
    

# ---------- Dataset ----------
class JsonlStudentDataset(Dataset):
    """
    Loads JSONL where each line is a dict with:
        - id: str
        - abstract: str   (abstract)
        - explanation: str  (teacher explanation)

    Returns items shaped for training:
        {"id": ..., "text": <formatted string>}
    """
    def __init__(self, path: str | Path,
                tokenizer,
                input_key: str = "abstract",
                output_key: str = "target_explanation",
                id_key: str = "id",
                *,
                strip_whitespace: bool = True,
                min_chars: int = 0,
                max_samples: Optional[int] = None):
        
        self.path = Path(path)
        self.input_key = input_key
        self.output_key = output_key
        self.id_key = id_key
        self.strip_whitespace = strip_whitespace
        self.min_chars = min_chars
        self.max_samples = max_samples
        
        self.samples: List[Sample] = []
        self._load()
        
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        s = self.samples[idx]
        return {"id": s.id, "text": self.format_text(s.abstract, s.target_explanation)}
    
    # def format_text(self, abstract:str, explanation:str) -> str:
    #     """ONE format the student trains on.

    #     Args:
    #         abstract (str): abstract
    #         explanation (str): explanation

    #     Returns:
    #         str: trainning sample format
    #     """
    #     return (
    #         "Task: Explain the abstract in simple, non-technical language. Stay strictly on-topic.\n\n"
    #         f"Abstract:\n{abstract}\n\nExplanation:\n{explanation}\n"
    #     )
    
    def format_text(self, abstract:str, explanation:str) -> str:
        # Notice: No space after the final colon in "Explanation:\n"
        return (
            f"{self.tokenizer.bos_token}Task: Explain the abstract in simple, non-technical language. "
            f"Stay strictly on-topic.\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Explanation:\n{explanation}{self.tokenizer.eos_token}"
        )

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if self.input_key not in data:
                    raise ValueError(f"Line {line_num}: '{self.input_key}' (abstract) missing")
                if self.id_key not in data:
                    raise ValueError(f"Line {line_num}: '{self.id_key}' missing")
                if self.output_key not in data:
                    raise ValueError(f"Line {line_num}: '{self.output_key}' (explanation) missing")

                abstract = data[self.input_key]
                explanation = data[self.output_key]
                sid = data[self.id_key]
                
                if self.strip_whitespace:
                    abstract = abstract.strip()
                    explanation = explanation.strip()
                    sid = str(sid).strip()
                self.samples.append(Sample(id=sid, abstract=abstract, target_explanation=explanation))
                if self.max_samples is not None and len(self.samples) >= self.max_samples:
                    break
