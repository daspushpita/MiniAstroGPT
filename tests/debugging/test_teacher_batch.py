from pathlib import Path
import sys
import time

start_time = time.time()
ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"

sys.path.insert(0, str(SRC))
from miniastrolm.llm.teacher import Llama_Teacher, Validation_Regeneration, Teacher_Data_Pipeline

# my_pipeline = Teacher_Data_Pipeline(database_path = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/processed/mini_astrolm.db"), 
#                                    output_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/test_data/teacher_output/v2"), 
#                                    prompt_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/teacher_prompt_v1.txt"),
#                                    batch_size=50, max_attempts = 3, raise_on_fail = False, log_skips=False, print_every=1,
#                                    max_total=500)

# teacher_data = my_pipeline.run()


my_pipeline = Teacher_Data_Pipeline(database_path = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/processed/mini_astrolm.db"), 
                                   output_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/test_data/teacher_output/v3"), 
                                   prompt_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/teacher_prompt_v2.txt"), 
                                   raise_on_fail = False, 
                                   log_skips=False, 
                                   print_every=1,
                                   max_total=4)

teacher_data = my_pipeline.run()
end_time = time.time()
execution_time = end_time - start_time

print(f"Executed in: {execution_time:.4f} seconds")