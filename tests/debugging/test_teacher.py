from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]      # project root (one level above tests/)
SRC  = ROOT / "src"

sys.path.insert(0, str(SRC))
from miniastrolm.llm.config import TeacherConfig
from miniastrolm.llm.llamacpp_teacher import LlamaCppTeacher
from miniastrolm.llm.teacher import Llama_Teacher
from miniastrolm.llm.validation_regeneration import Validation_Regeneration

my_config = TeacherConfig(
    n_ctx=4096,         # Perfect for abstracts + 800 word essays
    n_batch=512,        # Higher batch = faster prompt processing
    n_gpu_layers=-1,    # Offload everything
    n_threads=4,        # Use only the 4 "Performance" cores
    f16_kv=True,        # Ensure memory is saved as FP16
)

start_time = time.time()

def test_llm_teacher_init():
    teacher_model = LlamaCppTeacher(my_config=my_config)
    # teacher_model = Llama_Teacher(my_config=my_config, device='mps')

    validator = Validation_Regeneration(
        teacher_model=teacher_model,
        my_config = my_config,
        prompt_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/teacher/teacher_prompt_distill_v1.txt"),
        raise_on_fail=True,
        context_check_mode="off",
        judge_model=None)
    
    output = validator.generate_item(
        paper_id= "http://arxiv.org/abs/1801.00386v1",
        abstract = "We study the effects of cold dark matter on the propagation of gravitational waves of astrophysical and primordial origin. We show that the dominant effect of cold dark matter on gravitational waves from astrophysical sources is a small frequency dependent modification of the propagation speed of gravitational waves. However, the magnitude of the effect is too small to be detected in the near future. We furthermore show that the spectrum of primordial gravitational waves in principle contains detailed information about the properties of dark matter. However, depending on the wavelength, the effects are either suppressed because the dark matter is highly non-relativistic or because it contributes a small fraction of the energy density of the universe. As a consequence, the effects of cold dark matter on primordial gravitational waves in practice also appear too small to be detectable."
    )

    # assert teacher_model.model is not None, "Model not initialized."
    # assert teacher_model.tokenizer is not None, "Tokenizer not initialized."
    # assert teacher_model.device in ["cpu", "mps"], "Device should be either 'cpu' or 'mps'."
    return output
    
if __name__ == "__main__":
    my_model = test_llm_teacher_init()
    print(my_model)
    print("Llama_Teacher initialization test passed.")
    
end_time = time.time()
execution_time = end_time - start_time

print(f"Executed in: {execution_time:.4f} seconds")