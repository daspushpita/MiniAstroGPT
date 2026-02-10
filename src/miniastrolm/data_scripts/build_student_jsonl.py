from __future__ import annotations
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path

class TrainSampleGenerator:
    def __init__(self, judged_data_path, output_path_1,
                output_path_2: Path,
                write = True,
                max_samples: int = 10000):
        self.judged_data_path = judged_data_path
        self.output_path_1 = output_path_1 if output_path_1 else None
        self.output_path_2 = output_path_2 if output_path_2 else None
        self.max_samples = max_samples
        self.write = write
        self.accepted_samples = {}
        self.rejected_samples = {}
    
    def generate_train_samples(self):
        print("Generating training samples...")
        # ExitStack handles optional files gracefully
        with ExitStack() as stack:
            f_in = stack.enter_context(self.judged_data_path.open("r", encoding="utf-8"))
            
            # Only open output files if 'write' is True AND path is provided
            f_out_1 = stack.enter_context(self.output_path_1.open("w", encoding="utf-8")) if (self.write and self.output_path_1) else None
            f_out_2 = stack.enter_context(self.output_path_2.open("w", encoding="utf-8")) if (self.write and self.output_path_2) else None

            count = 0

            for line in f_in:
                if count >= self.max_samples:
                    break
                item = json.loads(line)
                if item.get("accepted"):
                    train_sample_id = item.get("id")
                    self.accepted_samples[train_sample_id] = item
                    if f_out_1: f_out_1.write(train_sample_id + "\n")
                    count += 1
                else:
                    train_sample_id = item.get("id")
                    self.rejected_samples[train_sample_id] = item
                    if f_out_2: f_out_2.write(train_sample_id + "\n")
        print(f"Generated training samples: {self.output_path_1}, {self.output_path_2}")
        return self.accepted_samples, self.rejected_samples
        
        
class Tranning_Samples_Split:
    def __init__(self, judged_samples_path, teacher_output_path,
                 output_path1, output_path2, output_path3, split_ratio,
                 data_format):
        self.judged_samples_path = judged_samples_path
        self.teacher_output_path = teacher_output_path
        self.output_path1 = output_path1
        self.output_path2 = output_path2
        self.output_path3 = output_path3
        self.accepted_samples, self.rejected_samples = TrainSampleGenerator(judged_data_path=self.judged_samples_path, 
                                                                            output_path_1=None, output_path_2=None, write=False).generate_train_samples()
        self.split_ratio = split_ratio
        self.data_format = data_format

    def read_write_data(self):
        
        teacher_lookup = {}
        print(f"Indexing teacher file: {self.teacher_output_path}")
        
        with self.teacher_output_path.open("r", encoding="utf-8") as f_teacher:
            for line in f_teacher:
                if not line.strip():
                    continue
                item = json.loads(line)
                t_id = item.get("id")
                if t_id:
                    teacher_lookup[t_id] = item
        
        n_total = len(teacher_lookup)
        n_train = int(self.split_ratio * n_total)
        n_train_f = int(self.split_ratio * n_train)
        n_val = int((1 - self.split_ratio) * n_train)
        n_test = int((1 - self.split_ratio) * n_total)
        
        # 3. Open ALL files at once to write continuously
        with self.output_path1.open("w", encoding="utf-8") as f_train, \
            self.output_path2.open("w", encoding="utf-8") as f_val, \
            self.output_path3.open("w", encoding="utf-8") as f_test:
                
                
            for i, (pid, judge_item) in enumerate(self.accepted_samples.items()):
                if i < n_train_f:
                    f_active = f_train
                elif n_train_f < i < n_train + n_val:
                    f_active = f_val
                else:
                    f_active = f_test
                    
                teacher_item = teacher_lookup[pid]
                if not teacher_item:
                    print(f"Warning: ID {pid} found in judge data but missing in teacher data.")
                    continue
                
                input = teacher_item.get('input','')
                title = teacher_item.get('title','')
                out = teacher_item.get("output", {})
                explanation = out.get("explanation", "") if isinstance(out, dict) else str(out)
                feedback = judge_item.get('judge', 'No feedback provided')

                # --- Formatting the Write ---
                if self.data_format=="txt":
                    f_active.write(f"## Sample ID: {pid}\n\n")
                    f_active.write(f"### Abstract:\n{input}\n\n")
                    f_active.write(f"### Generated Summary:\n{explanation}\n\n")
                    f_active.write(f"### Judge Feedback:\n{feedback}\n\n")
                    f_active.write("---\n\n")
                
                output_data = {
                                "id": pid,
                                "title":title,
                                "abstract": teacher_item.get("input",""),
                                "target_explanation": explanation,
                                "judge_feedback":feedback,
                                "accepted": True
                }
                f_active.write(json.dumps(output_data, ensure_ascii=False) + "\n")

        print(f"Success: {n_train_f} -> Train, {n_val} -> Val, {n_total - n_train_f - n_val} -> Test")        