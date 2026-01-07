import glob
import json
import re
from pathlib import Path


INPUT_PATTERN = "../raw/*.jsonl"
MERGED_PATH = Path("../processed/all_raw.jsonl")
CLEANED_PATH = Path("../processed/all_clean.jsonl")

CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)

# task 1: concat all jsonl files --------------------

with MERGED_PATH.open("w", encoding="utf-8") as fout:
    for path in sorted(glob.glob(INPUT_PATTERN)):
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    fout.write(line + "\n")

print(f"Concatenated into {MERGED_PATH}")

# tiny cleaner ---------------------------------------

def clean_text(t: str) -> str:
    if not t:
        return ""
    # remove inline LaTeX math like $...$
    t = re.sub(r"\$.*?\$", " ", t)
    # remove simple LaTeX commands like \alpha, \textbf{...}, \cite{...}
    t = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

# task 2: clean merged file -----------------------------

with MERGED_PATH.open("r", encoding="utf-8") as fin, \
     CLEANED_PATH.open("w", encoding="utf-8") as fout:

    for line in fin:
        obj = json.loads(line)
        if "title" in obj:
            obj["title_clean"] = clean_text(obj["title"])
        if "abstract" in obj:
            obj["abstract_clean"] = clean_text(obj["abstract"])
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Cleaned abstracts written to {CLEANED_PATH}")