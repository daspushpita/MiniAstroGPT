from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"

sys.path.insert(0, str(SRC))
from miniastrolm.llm.teacher import Llama_Teacher, Validation_Regeneration, Teacher_Data_Pipeline

my_pipeline = Teacher_Data_Pipeline(database_path = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/processed/mini_astrolm.db"), 
                                   output_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/test_data/teacher_output"), 
                                   prompt_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/teacher_prompt_v1.txt"),
                                   batch_size=5)

teacher_data = my_pipeline.run()