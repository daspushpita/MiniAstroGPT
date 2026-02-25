
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable
import json
from torch.utils.data import Dataset
from miniastrolm.student.prompting import build_full_text

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
        text = build_full_text(self.tokenizer, s.id, s.abstract, s.target_explanation)
        return {"id": s.id, "text": text}
    
    @staticmethod
    def _normalize_text(s: str) -> str:
        # Convert literal backslash escapes into real characters
        # (this is the important one for your dataset)
        s = s.replace("\\n", "\n")
        s = s.replace("\\t", "\t")
        s = s.replace("\r\n", "\n")
        s = s.replace("\u00a0", " ")
        #remove lines that look like they might be copyright notices, licenses, URLs, or DOIs
        bad = ("copyright", "license", "http", "www", "doi")
        kept = []
        for line in s.splitlines():
            l = line.lower()
            if any(b in l for b in bad):
                continue
            kept.append(line)
        return "\n".join(kept).strip()
    
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

                abstract = self._normalize_text(data[self.input_key])
                explanation = self._normalize_text(data[self.output_key])
                sid = data[self.id_key]
                
                if self.strip_whitespace:
                    abstract = abstract.strip()
                    explanation = explanation.strip()
                    sid = str(sid).strip()
                self.samples.append(Sample(id=sid, abstract=abstract, target_explanation=explanation))
                if self.max_samples is not None and len(self.samples) >= self.max_samples:
                    break
