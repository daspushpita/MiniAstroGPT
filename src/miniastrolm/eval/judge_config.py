from dataclasses import dataclass
from pathlib import Path

@dataclass
class JudgeConfig:
    """Hyperparameters + thresholds for judge behavior."""
    arch = "llama-cpp-python"
    max_attempts: int = 3
    max_new_tokens: int = 256
    max_new_tokens_retry: int = 512
    do_sample: bool = True
    temperature: float = 0.4
    top_p: float = 0.9
    repetition_penalty: float = 1.1    
    do_sample: bool = False
    
    min_faithfulness_to_keep: int = 4
    min_overall_to_keep: int = 60
    
    REQUIRED_TOP_KEYS = {"id", "scores", "error_tags", "rationale", "rewrite_hint"}
    REQUIRED_SCORE_KEYS = {"faithfulness", "clarity", "jargon", "structure", "uncertainty", "overall"}
    CANONICAL_TAGS = {"hallucination", "insufficient_coverage","style_drift", "overlap", "truncation", "non_json", "generic_filler"}