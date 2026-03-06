from __future__ import annotations
from datetime import date, datetime, timedelta
import os
import time
from pathlib import Path
import sqlite3

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
from functools import lru_cache

from miniastrolm.agent.explain import AstroAgent
from miniastrolm.cli.app import build_client

@dataclass(frozen=True)
class AgentConfig:
    provider: str
    max_turns: int
    max_revision_attempts: int
    threshold_hallucination: int
    threshold_clarity: int
    threshold_structure: int
    llama_model_path: Optional[Path] = None  


# === A) Daily data ===
def get_daily_paths_and_date():
    RAW_DIR = Path("data")  / "raw"
    PROCESSED_DIR = Path("data") / "processed"
    OUTPUT_DIR = Path("data") / "output"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    day_offset = int(1)#int(os.getenv("ASTRO_DAY_OFFSET", "0"))
    if day_offset < 0:
        raise ValueError("ASTRO_DAY_OFFSET must be >= 0.")
    today = date.today() - timedelta(days=day_offset)
    formatted_date = today.strftime("%Y%m%d")
    formatted_date_start = today.strftime("%Y%m%d000000")
    formatted_date_end = today.strftime("%Y%m%d235959")

    raw_path = RAW_DIR / f"arxiv_{formatted_date}.jsonl"
    merged_path = PROCESSED_DIR / f"merged_arxiv_{formatted_date}.jsonl"
    cleaned_path = PROCESSED_DIR / f"cleaned_arxiv_{formatted_date}.jsonl"
    db_output_path = PROCESSED_DIR / f"mini_astrolm_agentic_{formatted_date}.db"
    json_output_path = PROCESSED_DIR / f"mini_astrolm_agentic_{formatted_date}.jsonl"
    return raw_path, merged_path, cleaned_path, db_output_path, json_output_path, formatted_date_start, formatted_date_end

def run_fetch_and_build(date_from, date_to, raw_path: Path, merged_path: Path, 
                        cleaned_path: Path, db_output_path: Path, json_output_path: Path,
                        force: bool = False):
    from miniastrolm.daily.fetch import ArxivDownloader, Clean_Jsonl_Files
    from miniastrolm.db import SQLITE_Database_Builder

    # Validate arXiv date bounds (YYYYMMDDHHMMSS) and ordering.
    try:
        start_dt = datetime.strptime(str(date_from), "%Y%m%d%H%M%S")
        end_dt = datetime.strptime(str(date_to), "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError(
            "date_from/date_to must be in YYYYMMDDHHMMSS format."
        ) from exc
    if start_dt > end_dt:
        raise ValueError("date_from must be <= date_to.")

    run_ts = datetime.now().isoformat(timespec="seconds")
    # Normalize and prepare output directories.
    raw_path = Path(raw_path)
    merged_path = Path(merged_path)
    cleaned_path = Path(cleaned_path)
    db_output_path = Path(db_output_path)
    json_output_path = Path(json_output_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    db_output_path.parent.mkdir(parents=True, exist_ok=True)
    # json_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # First download the arXiv data for today's full day window.
    raw_exists = raw_path.exists() and raw_path.is_file() and raw_path.stat().st_size > 0
    if raw_exists and not force:
        fetch_action = "skipped"
    else:
        downloader = ArxivDownloader(date_from=date_from, date_to=date_to,
                                    outfile=raw_path).download()
        fetch_action = "rebuilt"

    #Clean the data
    merged_and_cleaned_exists = (
        merged_path.exists()
        and merged_path.is_file()
        and cleaned_path.exists()
        and cleaned_path.is_file()
        and cleaned_path.stat().st_size > 0
        and merged_path.stat().st_size > 0
    )
    # If raw data was rebuilt this run, downstream merged/cleaned artifacts must be rebuilt too.
    should_rebuild_merge_clean = force or (fetch_action == "rebuilt") or (not merged_and_cleaned_exists)
    if not should_rebuild_merge_clean:
        merge_and_clean_action = "skipped"
    else:
        cleaned_data = Clean_Jsonl_Files(INPUT_PATTERN = raw_path, 
                                        MERGED_PATH = merged_path,
                                        CLEANED_PATH = cleaned_path)
        cleaned_data.merge_inputs()
        cleaned_data.clean_merged_file()
        merge_and_clean_action = "rebuilt"

    #Build the sqlite database for the agent to query.
    db_exists = (db_output_path.exists() 
                and db_output_path.is_file() 
                and db_output_path.stat().st_size > 0)
    
    should_rebuild_db = force or should_rebuild_merge_clean or (not db_exists)
    if not should_rebuild_db:
        db_build_action = "skipped"
    else:
        SQLITE_Database_Builder(
            jason_file_path=cleaned_path,
            db_path=db_output_path,
        ).build_database()
        db_build_action = "rebuilt"

    return {
        "raw_path": raw_path,
        "merged_path": merged_path,
        "cleaned_path": cleaned_path,
        "db_path": db_output_path,
        "jsonl_path": json_output_path,
        "fetch_action": fetch_action,
        "merge_and_clean_action": merge_and_clean_action,
        "db_build_action": db_build_action,
        # "json_export_action": json_export_action,
        "status": "ok",
    }
    

# === B) DB queries ===
def get_random_papers(db_path: Path, n_samples: int = 1):
    
    query = """
            SELECT id, title_clean, abstract_clean
            FROM papers
            WHERE abstract_clean IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
            """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (n_samples,))
        rows = cursor.fetchall()
    return rows

# === C) Agent runner ===
def build_agent(config: AgentConfig):
    """
    Build LLM client + AstroAgent.
    """
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[TIMING] {ts} | client_creation | start")
    t0 = time.perf_counter()
    client = build_client(provider=config.provider, llama_model_path=config.llama_model_path)
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[TIMING] {ts} | client_creation | end | {time.perf_counter() - t0:.2f}s")
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[TIMING] {ts} | agent_creation | start")
    t0 = time.perf_counter()
    agent = AstroAgent(llm_client=client, max_turns=config.max_turns,
                    max_revision_attempts=config.max_revision_attempts,
                    threshold_hallucination=config.threshold_hallucination, 
                    threshold_clarity=config.threshold_clarity,
                    threshold_structure=config.threshold_structure)
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[TIMING] {ts} | agent_creation | end | {time.perf_counter() - t0:.2f}s")
    return agent

@lru_cache(maxsize=4)
def get_agent(config: AgentConfig) -> AstroAgent:
    return build_agent(config)

def run_agent_once(agent: AstroAgent, abstract: str, debug: bool):
    """
    Run the agent on a single abstract.
    """
    run_result = agent.run(abstract=abstract)

    if debug:
        for name in ["plan", "draft", "glossary", "critic", "revised_draft"]:
            if hasattr(run_result, name):
                print(f"\n==={name.upper()} ===\n{getattr(run_result, name)}")
    return run_result

def run_agent_batch(config: AgentConfig, 
                    agent: AstroAgent, 
                    batch_items: Iterable[Tuple[str,str,str]],
                    debug: bool = False, 
                    stop_on_error: bool = False):
    
    results = []
    errors = []
    traces = []
    for paper_id, title, abstract in batch_items:
        try:
            run_result = run_agent_once(agent, abstract, debug=debug)

            results.append({
                "id": paper_id,
                "title": title,
                "revised_draft": run_result.revised_draft,
            })

            if debug:
                traces.append({
                    "provider": config.provider,
                    "id": paper_id,
                    "title": title,
                    "plan": getattr(run_result, "plan", None),
                    "draft": getattr(run_result, "draft", None),
                    "glossary": getattr(run_result, "glossary", None),
                    "critic": getattr(run_result, "critic", None),
                    "revised_draft": getattr(run_result, "revised_draft", None),
                })
        
        except Exception as e:
            errors.append({
                "id": paper_id,
                "title": title,
                "error": str(e),
            })
            if stop_on_error:
                break
            continue
    if debug:
        return {"results": results, "errors": errors, "traces": traces}
    return {"results": results, "errors": errors}
        

def orchestration(config: AgentConfig,
                mode: str,
                abstract: Optional[str] = None,
                batch_items: Optional[Iterable[Tuple[str, str, str]]] = None,
                debug: bool = False,):
    
    my_agent = get_agent(config)
    
    if mode == "single":
        if abstract is None:
            raise ValueError("Mode is set to single but abstract not defined.")
        return run_agent_once(my_agent, abstract, debug)
    
    elif mode == "batch":
        if batch_items is None:
            raise ValueError("Mode is set to batch, but database not provided..")
        return run_agent_batch(config, my_agent, batch_items, debug=debug)

    else:
        raise ValueError("Mode is not defined..")
