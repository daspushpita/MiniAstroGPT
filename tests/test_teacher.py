from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"

sys.path.insert(0, str(SRC))
from miniastrolm.llm.teacher import Llama_Teacher, Validation_Regeneration

def test_llm_teacher_init():
    my_teacher = Llama_Teacher()
#     response = my_teacher.generate_response(prompt="""{"id": "http://arxiv.org/abs/1801.00386v1", "title_clean": "Gravitational Waves in Cold Dark Matter", "abstract_clean": "We study the effects of cold dark matter on the propagation of gravitational waves of astrophysical and primordial origin. We show that the dominant effect of cold dark matter on gravitational waves from astrophysical sources is a small frequency dependent modification of the propagation speed of gravitational waves. However, the magnitude of the effect is too small to be detected in the near future. We furthermore show that the spectrum of primordial gravitational waves in principle contains detailed information about the properties of dark matter. However, depending on the wavelength, the effects are either suppressed because the dark matter is highly non-relativistic or because it contributes a small fraction of the energy density of the universe. As a consequence, the effects of cold dark matter on primordial gravitational waves in practice also appear too small to be detectable."}
# """)
    
    validator = Validation_Regeneration(
        teacher_model=my_teacher,   # your existing Llama_Teacher instance
        max_attempts=3,
        prompt_path=Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/teacher_prompt_v1.txt"),
    )
    
    output = validator.generate_item(
        paper_id= "http://arxiv.org/abs/1801.00386v1",
        abstract = "We study the effects of cold dark matter on the propagation of gravitational waves of astrophysical and primordial origin. We show that the dominant effect of cold dark matter on gravitational waves from astrophysical sources is a small frequency dependent modification of the propagation speed of gravitational waves. However, the magnitude of the effect is too small to be detected in the near future. We furthermore show that the spectrum of primordial gravitational waves in principle contains detailed information about the properties of dark matter. However, depending on the wavelength, the effects are either suppressed because the dark matter is highly non-relativistic or because it contributes a small fraction of the energy density of the universe. As a consequence, the effects of cold dark matter on primordial gravitational waves in practice also appear too small to be detectable."
    )

    print(output["id"])
    print(len(output["explanation"]))
    print(output["explanation"][:500])
    assert my_teacher.model is not None, "Model not initialized."
    assert my_teacher.tokenizer is not None, "Tokenizer not initialized."
    assert my_teacher.device in ["cpu", "mps"], "Device should be either 'cpu' or 'mps'."
    return output
    
if __name__ == "__main__":
    my_model = test_llm_teacher_init()
    print(my_model)
    print("Llama_Teacher initialization test passed.")