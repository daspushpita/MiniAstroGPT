import sqlite3
import json
import os

jason_file_path = '/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/processed/all_clean.jsonl'

DB_PATH = os.path.join(os.path.dirname(jason_file_path), 'mini_astrolm.db')
#connect to a database (or create it if it doesn't exist)
conn = sqlite3.connect(DB_PATH)

#create a cursor object/ which is like a pen for the database
cur = conn.cursor()

#SQL commands
cur.execute('''CREATE TABLE papers (
    id TEXT PRIMARY KEY,
    year INTEGER,
    title TEXT,
    title_clean TEXT,
    abstract TEXT,
    abstract_clean TEXT
    )''')

with open(jason_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        data = json.loads(line)
        pid = data.get('id')
        year = data.get('year')
        title = data.get('title')
        title_clean = data.get('title_clean')
        abstract = data.get('abstract')
        abstract_clean = data.get('abstract_clean')
        
        cur.execute("""INSERT INTO papers (id, year, title, title_clean, abstract, abstract_clean)
        VALUES (?, ?, ?, ?, ?, ?)""", (pid, year, title, title_clean, abstract, abstract_clean))

conn.commit()
conn.close()

print("Done building SQLite DB:", DB_PATH)
