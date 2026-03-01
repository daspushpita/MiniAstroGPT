import json, argparse
from src.agent.astro_agent import AstroAgent
from src.data.arxiv_daily_fetch import ArxivDownloader, Clean_Jsonl_Files, SQLITE_Database_Builder
from src.cli.app import build_client, build_parser
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "agent_data" / "raw"
PROCESSED_DIR = ROOT / "agent_data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

today = date.today()
formatted_date = today.strftime("%Y%m%d")
formatted_date_start = today.strftime("%Y%m%d000000")
formatted_date_end = today.strftime("%Y%m%d235959")


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
    

def run_agent(agent: AstroAgent, abstract: str, debug: bool) -> None:
    run_result = agent.run(abstract=abstract)
    if debug:
        print(run_result.plan)
        print(run_result.draft)
        print(run_result.glossary)
        print(run_result.critic)
        print(run_result.revised_draft)

def main():

    parser = build_parser()
    parser.add_argument("--max-abstracts", type=int, default=None)
    parser.add_argument("--delete-raw", action="store_true")
    args = parser.parse_args()

    # Build today's cleaned JSONL. Set sql_database=True if needed for SQLite indexing.
    build_database(db_path=PROCESSED_DIR / f"mini_astrolm_agentic_{formatted_date}.db", sql_database=False)
    
    client = build_client(provider=args.provider, llama_model_path=args.llama_model_path)
    agent = AstroAgent(llm_client=client, max_turns=3)
    
    # Open the JSONL database and read records in one pass.
    with open(PROCESSED_DIR / f"cleaned_arxiv_{formatted_date}.jsonl", "r", encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin]
    
    #Initialize the agent and run it on the abstracts.
    records_to_run = records if args.max_abstracts is None else records[: max(0, args.max_abstracts)]
    for record in records_to_run:
        abstract = record.get("abstract_clean", "")
        title = record.get("title_clean", "")
        if not abstract:
            continue
        print(f"Running agent on abstract with title {title}...")
        run_agent(agent=agent, abstract=abstract, debug=True)

    if args.delete_raw:
        raw_path = RAW_DIR / f"arxiv_{formatted_date}.jsonl"
        if raw_path.exists():
            raw_path.unlink()
            print(f"Deleted raw file: {raw_path}")
        else:
            print(f"Raw file not found, nothing to delete: {raw_path}")


if __name__ == "__main__":
    main()
        

    
    
    
