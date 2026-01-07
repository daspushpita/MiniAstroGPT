import os, sys
from pathlib import Path
import sqlite3, json, tqdm

DB_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'mini_astrolm.db'
OUTPUT_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'batches'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PROMPT_PATH = Path(__file__).parent.parent.parent.parent / 'prompts' / 'teacher_prompt.txt'

# 1. connect to DB and sample rows ------------------------------------------------
database = sqlite3.connect(DB_PATH)
cur = database.cursor()

n_samples = 10000  # number of samples to create

cur.execute("""SELECT id, title_clean, abstract_clean FROM papers
                WHERE abstract_clean IS NOT NULL
                ORDER BY RANDOM()
                LIMIT ? """, (n_samples,))

rows = cur.fetchall()
database.close()

# 2. load prompt template ----------------------------------------------------------
with PROMPT_PATH.open("r", encoding="utf-8") as f:
    prompt_header = f.read().rstrip() + "\n\n"
    
# ---- 3. write batch file ----
batch_size = 50  # abstracts per batch file
n_batches = (len(rows) + batch_size - 1) // batch_size

for b in range(n_batches):
    start = b * batch_size
    end = min((b + 1) * batch_size, len(rows))
    out_path = OUTPUT_PATH / f"batch_{b+1:03d}.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(prompt_header)

        for i, row in enumerate(rows[start:end], start=1):
            pid, title, abstract = row
            f.write(f"--- ABSTRACT {i} ---\n")
            f.write(f"ID: {pid}\n")
            f.write(f"TITLE: {title}\n")
            f.write("ABSTRACT:\n")
            f.write(abstract + "\n\n")

print(f"Done creating {n_batches} batch files at:", OUTPUT_PATH)
        
    
