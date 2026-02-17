from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"

sys.path.insert(0, str(SRC))
from miniastrolm.llm.config import TeacherConfig
from miniastrolm.llm.llamacpp_teacher import LlamaCppTeacher
from miniastrolm.llm.validation_regeneration import Teacher_Data_Pipeline

start_time = time.time()

my_config = TeacherConfig(
    n_ctx=4096,
    n_batch=512,
    n_gpu_layers=-1,
    n_threads=4,
    f16_kv=True,
)
teacher_model = LlamaCppTeacher(my_config=my_config)
my_pipeline = Teacher_Data_Pipeline(database_path = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/processed/mini_astrolm.db"), 
                                    output_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/teacher/v1"), 
                                    prompt_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/teacher/teacher_prompt_distill_v1.txt"),
                                    teacher_model = teacher_model,
                                    my_config=my_config,
                                    raise_on_fail = False, 
                                    log_skips=False, 
                                    print_every=25,
                                    max_accepted=200,
                                    exclude_ids_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/teacher/v1/eval_v1_ids.txt"))

teacher_data = my_pipeline.run()
end_time = time.time()
execution_time = end_time - start_time

print(f"Executed in: {execution_time:.4f} seconds")
