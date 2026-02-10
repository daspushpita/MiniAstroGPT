from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from miniastrolm.data_scripts.generate_train_samples import Tranning_Samples_Split
JUDGED_DATA_PATH = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/evals/judge_results_v1.jsonl")
TEACHER_DATA_PATH = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/teacher_output/v2/train.jsonl")

OUTPUT_PATH_1 = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/judged_samples/train.jsonl")
OUTPUT_PATH_2 = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/judged_samples/validation.jsonl")
OUTPUT_PATH_3 = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/judged_samples/test.jsonl")

# generator = TrainSampleGenerator(JUDGED_DATA_PATH, OUTPUT_PATH_1, OUTPUT_PATH_2).generate_train_samples()
generator = Tranning_Samples_Split(judged_samples_path=JUDGED_DATA_PATH,
                                   teacher_output_path=TEACHER_DATA_PATH,
                                   output_path1=OUTPUT_PATH_1,
                                   output_path2=OUTPUT_PATH_2,
                                   output_path3=OUTPUT_PATH_3,
                                   split_ratio=0.9,
                                   data_format="jsonl").read_write_data()
