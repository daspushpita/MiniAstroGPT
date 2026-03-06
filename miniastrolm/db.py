import sqlite3
import requests, time, json, xml.etree.ElementTree as ET
import glob, re
from pathlib import Path
import pandas as pd

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
