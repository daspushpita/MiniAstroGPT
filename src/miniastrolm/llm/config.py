from dataclasses import dataclass
from pathlib import Path

# DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B", but we use the instruct-tuned version for better explanations
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

@dataclass
class TeacherConfig:
    model_id: str = DEFAULT_MODEL_ID        # Hugging Face model ID or local path for the teacher model
    four_bit_teacher: bool = True           # Whether to use 4-bit quantization for the teacher model (saves memory, may reduce quality)
    architecture = "llama-cpp-python"       #llama_hf, llama-cpp-python etc.
    max_new_tokens: int = 768
    max_new_tokens_retry: int = 1024
    do_sample: bool = True
    temperature: float = 0.2
    top_p: float = 0.8
    repetition_penalty: float = 1.15

    min_chars: int = 900
    max_attempts: int = 3
    context_min_coverage: float = 0.6
    device="mps"

    # pipeline
    data_batch_size: int = 100
    llm_batch_size: int = 1
    llm_batch: bool = False

    # llama.cpp-only (ignored by HF)
    model_path: Path = Path("/Users/pushpita/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = -1
    n_batch: int = 512        # Higher batch = faster initial prompt reading
    f16_kv: bool = True        # Compresses memory to half-precision
    seed: int = 42
    flash_attn: bool = True
    verbose: bool = False