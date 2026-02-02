from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

class TrainSampleGenerator:
    def __init__(self, judged_data_path: Path, output_path_1: Path,
                output_path_2: Path,
                max_samples: int = 10000):
        self.judged_data_path = judged_data_path
        self.output_path_1 = output_path_1
        self.output_path_2 = output_path_2
        self.max_samples = max_samples
    
    def generate_train_samples(self):
        print("Generating training samples...")
        with self.judged_data_path.open("r", encoding="utf-8") as f_in, \
            self.output_path_1.open("w", encoding="utf-8") as f_out_1, \
            self.output_path_2.open("w", encoding="utf-8") as f_out_2:
            print("exists:", self.output_path_1.exists(), "size:", self.output_path_1.stat().st_size if self.output_path_1.exists() else None, flush=True)
            print("exists:", self.output_path_2.exists(), "size:", self.output_path_2.stat().st_size if self.output_path_2.exists() else None, flush=True)
            count = 0
            for line in f_in:
                if count >= self.max_samples:
                    break
                item = json.loads(line)
                if item.get("accepted"):
                    train_sample_id = item.get("id")
                    f_out_1.write(train_sample_id + "\n")
                    count += 1
                else:
                    train_sample_id = item.get("id")
                    f_out_2.write(train_sample_id + "\n")
        print(f"Generated training samples: {self.output_path_1}, {self.output_path_2}")
                    
        