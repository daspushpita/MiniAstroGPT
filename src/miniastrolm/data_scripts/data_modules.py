import requests, time, json, xml.etree.ElementTree as ET
import numpy as np
import os, glob, re
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
import sqlite3


class ArxivDownloader:
    
    def __init__(
        self,
        *,
        date_from: str,
        date_to: str,
        outfile: Path,
        max_results: int = 500,
        categories: list[str] | None = None,
        sleep_seconds: int = 10):
        
        
        self.date_from = date_from
        self.date_to = date_to
        self.outfile = Path(outfile)
        self.max_results = max_results
        self.sleep_seconds = sleep_seconds
        # ---------------------------
        # PARAMETERS
        # ---------------------------
        # Known astro-ph categories (explicit OR instead of wildcard)
        self.ASTRO_CATS = ["astro-ph", "astro-ph.HE", "astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
              "astro-ph.IM", "astro-ph.SR"]
        self.CAT_QUERY = "(" + " OR ".join(f"cat:{c}" for c in self.ASTRO_CATS) + ")"


    def get_text(self, element, default=""):
        """Helper function to get text from the XML element.
        """
        if element is not None and element.text is not None:
            return element.text.strip().replace("\n", " ")
        return default

    # ---------------------------
    # MAIN LOOP
    # ---------------------------

    def download(self, *args, **kwargs):

        base_url = "https://export.arxiv.org/api/query"
        HEADERS = {
            "User-Agent": "AstroGPT-arxiv-scraper/0.1 (pushpitads1996@gmail.com)"
        }

        if self.outfile.exists():
            print(f"Output file {self.outfile} already exists. Appending...")
            existing_count = sum(1 for _ in self.outfile.open("r", encoding="utf-8"))
            start = existing_count
            count = existing_count
            mode = "a"
            print(f"Resuming from record {start}...")
        else:
            self.outfile.parent.mkdir(parents=True, exist_ok=True)
            start = 0
            count = 0
            mode = "w"
            print(f"Creating new output file {self.outfile}...")

        total_results = None

        with self.outfile.open(mode, encoding="utf-8") as fout:
            while True:
                print(f"Fetching results {start} to {start + self.max_results}...")
                params = {
                    "search_query": f"{self.CAT_QUERY} AND submittedDate:[{self.date_from} TO {self.date_to}]",
                    "start": start,
                    "max_results": self.max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "ascending"
                }
                # Send the request to arXiv’s server.
                # This line actually "asks" arXiv for the data.
                print(f"Fetching {start}-{start+self.max_results}...")
                
                try:
                    response = requests.get(base_url, params=params, headers=HEADERS, timeout=60)
                    response.raise_for_status()
                except requests.RequestException as e:
                    print(f"Request failed at start={start}: {e}")
                    break

                print("URL sent:", response.url)
                # Parse the XML response
                
                try:
                    root = ET.fromstring(response.text)
                except ET.ParseError as e:
                    print(f"XML parse error at start={start}: {e}")
                    break 

                ns = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "opensearch": "http://a9.com/-/spec/opensearch/1.1/"
                }
                if total_results is None and start == 0:
                    tr = root.find("opensearch:totalResults", ns)
                    if tr is not None and tr.text is not None:
                        total_results = int(tr.text.strip())
                        print(f"arXiv reports total_results = {total_results}")

                entries = root.findall("atom:entry", ns)
                print("entries on this page:", len(entries))

                if not entries:
                    print("No more entries.")
                    break        
                for e in entries:
                    title = self.get_text(e.find("atom:title", ns))
                    summary = self.get_text(e.find("atom:summary", ns))
                    paper_id = self.get_text(e.find("atom:id", ns))
                    published = self.get_text(e.find("atom:published", ns))
                    cats = [c.attrib.get("term", "") for c in e.findall("atom:category", ns)]
                    
                    record = {"id": paper_id,
                                "title": title,
                                "abstract": summary,
                                "published": published,
                                "categories": cats}
                    
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                if len(entries) < self.max_results:
                    break
                start += self.max_results
                time.sleep(self.sleep_seconds)  # be polite to arXiv's servers
                
        print(f"Downloaded {count} abstracts to {self.outfile}.")
            
            
class Clean_Jsonl_Files:
    
    """Generating cleaned jsonl files
    """
    def __init__(self, INPUT_PATTERN="../raw/*.jsonl", 
                 MERGED_PATH=Path("../processed/all_raw.jsonl"), 
                 CLEANED_PATH=Path("../processed/all_clean.jsonl")):
        self.INPUT_PATTERN = INPUT_PATTERN
        self.MERGED_PATH = MERGED_PATH
        self.CLEANED_PATH = CLEANED_PATH
        self.CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
        
    def merge_inputs(self):
        with self.MERGED_PATH.open("w", encoding="utf-8") as fout:
            for path in sorted(glob.glob(self.INPUT_PATTERN)):
                with open(path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            fout.write(line + "\n")

        print(f"Concatenated into {self.MERGED_PATH}")
        return fout

    def clean_text(self, t: str) -> str:
        if not t:
            return ""
        # remove inline LaTeX math like $...$
        t = re.sub(r"\$.*?\$", " ", t)
        # remove simple LaTeX commands like \alpha, \textbf{...}, \cite{...}
        t = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def clean_merged_file(self):
        with self.MERGED_PATH.open("r", encoding="utf-8") as fin, \
             self.CLEANED_PATH.open("w", encoding="utf-8") as fout:

            for line in fin:
                obj = json.loads(line)
                if "title" in obj:
                    obj["title_clean"] = self.clean_text(obj["title"])
                if "abstract" in obj:
                    obj["abstract_clean"] = self.clean_text(obj["abstract"])
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    
class SQLITE_Database_Builder:
    
    def __init__(self, jason_file_path: str | Path, db_path: str | Path | None = None):
        self.jason_file_path = Path(jason_file_path)
        if db_path is None:
            db_path = self.jason_file_path.parent / "mini_astrolm.db"
        self.db_path = Path(db_path)
        
    def build_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute('''CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    year INTEGER,
                    title TEXT,
                    title_clean TEXT,
                    abstract TEXT,
                    abstract_clean TEXT
                    )''')
        
        with open(self.jason_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                pid = data.get('id')
                # year = data.get('year')
                published = data.get("published")
                year = int(published[:4]) if published else None
                title = data.get('title')
                title_clean = data.get('title_clean')
                abstract = data.get('abstract')
                abstract_clean = data.get('abstract_clean')
                
                cur.execute("""INSERT OR IGNORE INTO papers (id, year, title, title_clean, abstract, abstract_clean)
                VALUES (?, ?, ?, ?, ?, ?)""", (pid, year, title, title_clean, abstract, abstract_clean))
                
                
        conn.commit()
        conn.close()
        print("Done building SQLite DB:", self.db_path)

    def generate_batches(self, n_samples=10000, batch_size=50,
                         db_path = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'mini_astrolm.db',
                         output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'batches',
                         prompt_path = Path(__file__).parent.parent.parent.parent / 'prompts' / 'teacher_prompt.txt'):
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.output_path = output_path
        self.prompt_path = prompt_path 
        
        database = sqlite3.connect(self.db_path)
        cur = database.cursor()
        cur.execute("""SELECT id, title_clean, abstract_clean from papers
                        where abstract_clean is not null
                        ORDER BY RANDOM()
                        LIMIT ?""", (n_samples,))
        
        rows = cur.fetchall()
        database.close()
        
        with self.prompt_path.open("r", encoding="utf-8") as f:
            prompt_header = f.read().rstrip() + "\n\n"
            
        n_batches = len(rows) // batch_size + (1 if len(rows) % batch_size != 0 else 0)
        
        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, len(rows))
            out_path = self.output_path / f"batch_{b+1:03d}.txt"
            
            with out_path.open("w", encoding="utf-8") as f:
                f.write(prompt_header)
                
                for i, row in enumerate(rows[start:end], start=1):
                    pid, title, abstract = row
                    f.write(f"--- ABSTRACT {i} ---\n")
                    f.write(f"ID: {pid}\n")
                    f.write(f"TITLE: {title}\n")
                    f.write("ABSTRACT:\n")
                    f.write(abstract + "\n\n")
        print(f"Done creating {n_batches} batch files at:", self.output_path)


class split_data_class:
    
    def __init__(self, input_jason, split_ratio, seed=42):
        self.input_jason = input_jason
        self.split_ratio = split_ratio
        self.seed = seed
        
    def split_data(self):
        """Splits the data into train and validation sets

        Args:
            input_jason (_type_): input jason file path
            split_ratio (_type_): ratio for train and validation split
            seed (int, optional): _description_. Defaults to 42.
        """
        df = pd.read_json(self.input_jason, lines=True, orient="records")
        train_df, val_df = train_test_split(df, test_size=self.split_ratio, random_state=self.seed)

        train_df.to_json("train_data.jsonl", lines=True, orient="records", force_ascii=False)
        val_df.to_json("val_data.jsonl", lines=True, orient="records", force_ascii=False)
        print(f"Train: {len(train_df)}, Val: {len(val_df)} saved to JSONL files")
        return train_df, val_df

class jason_to_txt:
    
    def __init__(self, input_jason_file, output_txt_file):
        self.jason_file = input_jason_file
        self.output_txt_file = output_txt_file
        
    def convert(self):
        df = pd.read_json(self.jason_file, lines=True, orient="records")
        print(f"Loaded {len(df)} entries from {self.jason_file}")
        
        with open(self.output_txt_file, "w", encoding="utf-8") as fout:
            for abstract in df["abstract"]:
                abstract = abstract.strip().replace("\n", " ")
                fout.write(abstract + "\n\n<eos>\n\n")  # separate samples
        print(f"Saved abstracts to {self.output_txt_file}")
        return self.output_txt_file
    
class convert_sqlite_to_jasonl:
    
    def __init__(self, limit, offset, input_data_path, output_jason_file,*args, **kwargs):
        self.limit = limit
        self.offset = offset
        self.input_data_path = input_data_path
        self.conn = sqlite3.connect(self.input_data_path)
        self.output_jason_file = output_jason_file
        
    def fetch_data(self):
        cur = self.conn.cursor()
        cur.execute(""" SELECT id, title_clean, abstract_clean
                    FROM papers
                    WHERE trim(abstract_clean) != '' AND trim(title_clean) != ''
                    ORDER BY id
                    LIMIT ? OFFSET ? """, (self.limit, self.offset))
        
        return cur.fetchall()
    
    def save_to_jsonl(self):
        rows = self.fetch_data()
        self.output_jason_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_jason_file.open("w", encoding ="utf-8") as fout:
            for row in rows:
                pid, title_clean, abstract_clean = row
                record = {
                    "id": pid,
                    "title_clean": title_clean,
                    "abstract_clean": abstract_clean
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(rows)} records to {self.output_jason_file}")
        self.conn.close()
        return self.output_jason_file
    
# class Distillation_pipeline:
    
#     def __init__(self, sqlite_db_path, output_dir, batch_size=50, *args, **kwargs):
#         self.sqlite_db_path = sqlite_db_path
#         self.output_dir = Path(output_dir)
#         self.batch_size = batch_size
#         self.output_dir.mkdir(parents=True, exist_ok=True)
    
#     def count_categories(self):
#         conn = sqlite3.connect(self.sqlite_db_path)
#         cur = conn.cursor()
#         cur.execute()
        