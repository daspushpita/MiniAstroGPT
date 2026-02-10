from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"

sys.path.insert(0, str(SRC))
from miniastrolm.llm.config import TeacherConfig
from miniastrolm.llm.llamacpp_teacher import LlamaCppTeacher
from miniastrolm.llm.validation_regeneration import Teacher_Data_Pipeline

# This block of code is creating an instance of the `TeacherConfig` class with specific configuration
# settings for a teacher model. Here's a breakdown of what each parameter is doing:
my_config = TeacherConfig(
    n_ctx=4096,         # Perfect for abstracts + 800 word essays
    n_batch=512,        # Higher batch = faster prompt processing
    n_gpu_layers=-1,    # Offload everything
    n_threads=4,        # Use only the 4 "Performance" cores
    f16_kv=True,        # Ensure memory is saved as FP16
)
teacher_model = LlamaCppTeacher(my_config=my_config)
my_pipeline = Teacher_Data_Pipeline(database_path = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/processed/mini_astrolm.db"), 
                                    output_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/teacher_output/v2/"), 
                                    prompt_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/teacher_prompt_v1_final.txt"),
                                    teacher_model = teacher_model,
                                    my_config=my_config,
                                    raise_on_fail = False, 
                                    log_skips=False, 
                                    print_every=1,
                                    max_total=1000)

teacher_data = my_pipeline.run()
end_time = time.time()
execution_time = end_time - start_time

print(f"Executed in: {execution_time:.4f} seconds")