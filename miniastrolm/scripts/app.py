from __future__ import annotations
import os
from pathlib import Path
# import gradio as gr
import time


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
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default

# --- CONFIG: set these once ---
CFG = AgentConfig(
    provider=os.getenv("ASTRO_PROVIDER", "openai"),
    num_days=3,
    max_turns=3,
    max_revision_attempts=1,
    threshold_hallucination=2,
    threshold_clarity=2,
    threshold_structure=2,
    llama_model_path=os.getenv("LLAMA_MODEL_PATH"),
    fast_mode=_env_flag("ASTRO_FAST_MODE", "1"),
    fast_max_new_tokens=_env_int("ASTRO_FAST_MAX_NEW_TOKENS", 700),
)

start_time = time.time()
def refresh_db(force: bool = False) -> str:
    try:
        raw_path, merged_path, cleaned_path, db_output_path, json_output_path, cache_path, date_from, date_to = get_daily_paths_and_date(CFG)
        info = run_fetch_and_build(date_from=date_from, date_to=date_to, 
                                    raw_path = raw_path, merged_path=merged_path, 
                                    cleaned_path=cleaned_path, db_output_path=db_output_path, 
                                    json_output_path=json_output_path,
                                    force = force)
        return f"DB status: {info['status']} | fetch={info['fetch_action']} clean={info['merge_and_clean_action']} db={info['db_build_action']}\nDB: {info['db_path']}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def load_random() -> tuple[str, str, str]:
    try:
        raw_path, merged_path, cleaned_path, db_output_path, json_output_path, cache_path, date_from, date_to = get_daily_paths_and_date(CFG)
        rows = get_random_papers(db_output_path, n_samples=1)
        if not rows:
            return "", "No papers available", "Refresh papers to build/populate the daily DB first."
        paper_id, title, abstract = rows[0]
        return paper_id, title, abstract, cache_path
    except Exception as e:
        return "", f"Error: {type(e).__name__}", str(e)


def explain(paper_id: str, abstract: str, cache_path:Path, debug: bool):
    if not str(abstract).strip():
        return "Provide or load an abstract first."
    try:
        run_result = orchestration(
            config=CFG,
            mode="single",
            abstract=abstract,
            debug=debug,
            cache_path=cache_path
        )
        final_text = getattr(run_result, "revised_draft", None) or str(run_result)
        return final_text
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def main():
    cleanup_after_run = os.getenv("ASTRO_CLEANUP_AFTER_RUN", "1").strip().lower() in {"1", "true", "yes", "on"}
    raw_path, merged_path, cleaned_path, db_output_path, json_output_path, _cache_path, _date_from, _date_to = get_daily_paths_and_date(CFG)
    try:
        my_data = refresh_db()
        paper_id, title, abstract, cache_path = load_random()
        final_text = explain(paper_id, abstract, cache_path=cache_path, debug=True)
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

end_time = time.time()
execution_time = end_time - start_time

print(f"Executed in: {execution_time:.4f} seconds")

# # --- UI ---
# with gr.Blocks() as demo:
#     gr.Markdown("# AstroGPT (Daily Astronomy arXiv)")
#     state_paper_id = gr.State("")
#     with gr.Row():
#         btn_refresh = gr.Button("Refresh papers (build DB)")
#         force_refresh = gr.Checkbox(value=False, label="Force rebuild")
#     refresh_out = gr.Textbox(label="Refresh status", lines=2)
    
#     with gr.Row():
#         btn_random = gr.Button("Random paper")
#         debug = gr.Checkbox(value=False, label="Debug")

#     title_box = gr.Textbox(label="Title", lines=2)
#     abstract_box = gr.Textbox(label="Abstract", lines=10)

#     btn_explain = gr.Button("Explain")
#     out_box = gr.Textbox(label="Explanation", lines=12)
    
#     btn_refresh.click(refresh_db, inputs=[force_refresh], outputs=[refresh_out])
#     btn_random.click(load_random, inputs=None, outputs=[state_paper_id, title_box, abstract_box])
#     btn_explain.click(explain, inputs=[state_paper_id, abstract_box, debug], outputs=[out_box])

# demo.launch(show_error=True)
        
