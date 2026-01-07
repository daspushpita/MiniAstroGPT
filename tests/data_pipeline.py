from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"

sys.path.insert(0, str(SRC))

from miniastrolm.data_scripts.data_modules import (
    ArxivDownloader,
    Clean_Jsonl_Files,
    SQLITE_Database_Builder,
)

def test_data_pipeline():
    
    RAW_DIR = Path("../test_data/mini_astrolm/raw")
    PROCESSED_DIR = Path("../test_data/mini_astrolm/processed")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # downloader = ArxivDownloader(
    #     date_from="202501010000",
    #     date_to="202502010000",
    #     outfile= PROCESSED_DIR / "data/raw/astro_abstracts_2025.jsonl",
    #     max_results=10,
    # )
    # downloader.download()
    # assert any(downloader.output_dir.glob("*.jsonl")), "No files downloaded."

    cleaner = Clean_Jsonl_Files(INPUT_PATTERN = "../test_data/mini_astrolm/processed/*.jsonl",
                                MERGED_PATH=Path("../test_data/mini_astrolm/processed/all_raw.jsonl"),
                                CLEANED_PATH=Path("../test_data/mini_astrolm/processed/all_clean.jsonl"))
    cleaner.merge_inputs()
    cleaner.clean_merged_file()

    db_builder = SQLITE_Database_Builder(
        jason_file_path=Path("../test_data/mini_astrolm/processed/all_clean.jsonl"),
        db_path=Path("../test_data/mini_astrolm/mini_astrolm.db"),
    )
    
    db_builder.build_database()
    assert db_builder.db_path.exists(), "Database file not created."
    
if __name__ == "__main__":
    test_data_pipeline()