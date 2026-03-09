import os
from pathlib import Path

from miniastrolm.scripts.utils import (
    AgentConfig,
    get_daily_paths_and_date,
    run_fetch_and_build,
    get_random_papers,
    orchestration,
    cleanup_session_artifacts,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


# --- CONFIG ---
my_config = AgentConfig(
    provider=os.getenv("ASTRO_PROVIDER", "llama"),
    num_days = 3,
    max_turns=3,
    max_revision_attempts=1,
    threshold_hallucination=2,
    threshold_clarity=2,
    threshold_structure=2,
    llama_model_path=Path("/Users/pushpita/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    fast_mode=False,
    fast_max_new_tokens=1024,
    write_mode = True
)

def main():
    cleanup_after_run = os.getenv("ASTRO_CLEANUP_AFTER_RUN", "1").strip().lower() in {"1", "true", "yes", "on"}
    raw_path, merged_path, cleaned_path, db_output_path, json_output_path, cache_path, date_from, date_to = get_daily_paths_and_date(my_config)
    try:
        info = run_fetch_and_build(date_from=date_from, date_to=date_to, 
                                    raw_path = raw_path, merged_path=merged_path, 
                                    cleaned_path=cleaned_path, db_output_path=db_output_path, 
                                    json_output_path=json_output_path,
                                    force = False)
        all_abstracts = get_random_papers(db_output_path, n_samples=2)
        if not all_abstracts:
            return "", "No papers available", "Refresh papers to build/populate the daily DB first."
            
        final_text = orchestration(my_config,
                                    mode="batch",
                                    batch_items=all_abstracts,
                                    debug=False,
                                    cache_path=cache_path)
        print(final_text)
        return final_text
    finally:
        if cleanup_after_run:
            cleanup_session_artifacts(
                raw_path=raw_path,
                merged_path=merged_path,
                cleaned_path=cleaned_path,
                db_output_path=db_output_path,
                json_output_path=json_output_path,
                extra_dirs=[raw_path.parent.parent / "output"],
            )

if __name__ == "__main__":
    main()
