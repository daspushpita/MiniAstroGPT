import json, argparse
from miniastrolm.agent.explain import AstroAgent
from miniastrolm.daily.fetch import ArxivDownloader, Clean_Jsonl_Files
from miniastrolm.scripts.db import SQLITE_Database_Builder
from miniastrolm.cli.app import build_client, build_parser
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "agent_data" / "raw"
PROCESSED_DIR = ROOT / "agent_data" / "processed"
OUTPUT_DIR = ROOT / "agent_data" / "output"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

today = date.today()
formatted_date = today.strftime("%Y%m%d")
formatted_date_start = today.strftime("%Y%m%d000000")
formatted_date_end = today.strftime("%Y%m%d235959")

# formatted_date_start = today.strftime("20260224000000")
# formatted_date_end = today.strftime("20260224235959")


def build_database(db_path: str, sql_database: bool) -> None:
    """Builds a sqlite database of the daily abstracts from arxiv...

    Args:
        db_path (str): The path to the database file.
    """
    
    # First download the arXiv data for today's full day window.
    raw_path = RAW_DIR / f"arxiv_{formatted_date}.jsonl"
    merged_path = PROCESSED_DIR / f"merged_arxiv_{formatted_date}.jsonl"
    cleaned_path = PROCESSED_DIR / f"cleaned_arxiv_{formatted_date}.jsonl"
    
    downloader = ArxivDownloader(date_from=formatted_date_start, date_to=formatted_date_end,
                                outfile=raw_path)
    downloader.download()
    
    #Clean the data
    cleaned_data = Clean_Jsonl_Files(INPUT_PATTERN = raw_path, 
                                    MERGED_PATH = merged_path,
                                    CLEANED_PATH = cleaned_path)
    cleaned_data.merge_inputs()
    cleaned_data.clean_merged_file()
    
    #Optionally build the sqlite database for the agent to query.
    if sql_database:
        db_builder = SQLITE_Database_Builder(
            jason_file_path=cleaned_path,
            db_path=PROCESSED_DIR / f"mini_astrolm_agentic_{formatted_date}.db",
        )
        
        db_builder.build_database()
        assert db_builder.db_path.exists(), "Database file not created."

    return "Database build complete at " + str(cleaned_path)
    

def run_agent(agent: AstroAgent, abstract: str, debug: bool):
    run_result = agent.run(abstract=abstract)
    if debug:
        print(run_result.plan)
        print(run_result.draft)
        print(run_result.glossary)
        print(run_result.critic)
        print(run_result.revised_draft)
    return run_result

def main():

    parser = build_parser()
    parser.add_argument("--max-abstracts", type=int, default=None)
    parser.add_argument("--delete-raw", action="store_true")
    args = parser.parse_args()

    # Build today's cleaned JSONL. Set sql_database=True if needed for SQLite indexing.
    build_database(db_path=PROCESSED_DIR / f"mini_astrolm_agentic_{formatted_date}.db", sql_database=False)
    
    client = build_client(provider=args.provider, llama_model_path=args.llama_model_path)
    agent = AstroAgent(llm_client=client, max_turns=3,max_revision_attempts=3,
                        threshold_hallucination=1, 
                        threshold_clarity=2,
                        threshold_structure=2)
    
    # Open the JSONL database and read records in one pass.
    with open(PROCESSED_DIR / f"cleaned_arxiv_{formatted_date}.jsonl", "r", encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin]

    output_path = OUTPUT_DIR / f"agent_runs_{formatted_date}.jsonl"
    md_output_path = OUTPUT_DIR / f"agent_runs_{formatted_date}.md"
    run_ts = datetime.now().isoformat(timespec="seconds")
    print(f"Writing agent outputs to: {output_path}")
    print(f"Writing human-readable report to: {md_output_path}")
    
    #Initialize the agent and run it on the abstracts.
    records_to_run = records if args.max_abstracts is None else records[: max(0, args.max_abstracts)]
    with output_path.open("a", encoding="utf-8") as fout, md_output_path.open("a", encoding="utf-8") as mdout:
        if md_output_path.stat().st_size == 0:
            mdout.write(f"# Agent Runs - {formatted_date}\n\n")
        for record in records_to_run:
            paper_id = record.get("id", "")
            title = record.get("title_clean", "")
            abstract = record.get("abstract_clean", "")
            if not abstract:
                continue
            print(f"Running agent on abstract with title {title}...")
            run_result = run_agent(agent=agent, abstract=abstract, debug=False)
            out_row = {
                "run_ts": run_ts,
                "provider": args.provider,
                "id": paper_id,
                "title": title,
                "mode": run_result.mode,
                "plan": run_result.plan,
                "draft": run_result.draft,
                "glossary": run_result.glossary,
                "critic": run_result.critic,
                "revised_draft": run_result.revised_draft,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            mdout.write(f"## {title or 'Untitled'}\n\n")
            mdout.write(f"- Run Timestamp: `{run_ts}`\n")
            mdout.write(f"- Provider: `{args.provider}`\n")
            mdout.write(f"- ID: `{paper_id}`\n")
            mdout.write(f"- Mode: `{run_result.mode}`\n\n")
            mdout.write("### Plan\n\n")
            mdout.write(f"{run_result.plan}\n\n")
            mdout.write("### Draft\n\n")
            mdout.write(f"{run_result.draft}\n\n")
            mdout.write("### Glossary\n\n")
            mdout.write(f"{run_result.glossary}\n\n")
            mdout.write("### Critic\n\n")
            mdout.write(f"{run_result.critic}\n\n")
            mdout.write("### Revised Draft\n\n")
            mdout.write(f"{run_result.revised_draft}\n\n")
            mdout.write("---\n\n")

    if args.delete_raw:
        raw_path = RAW_DIR / f"arxiv_{formatted_date}.jsonl"
        if raw_path.exists():
            raw_path.unlink()
            print(f"Deleted raw file: {raw_path}")
        else:
            print(f"Raw file not found, nothing to delete: {raw_path}")


if __name__ == "__main__":
    main()
        

    
    
    
