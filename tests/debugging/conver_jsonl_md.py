import json
from pathlib import Path

INPUT_JSONL = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/test_data/teacher_output/train.jsonl")     # change this
OUTPUT_DIR = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/test_data/teacher_output/")
OUTPUT_MD   = OUTPUT_DIR / "inspection.md"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_JSONL, "r", encoding="utf-8") as f, \
     open(OUTPUT_MD, "w", encoding="utf-8") as md:

    for i, line in enumerate(f, start=1):
        if not line.strip():
            continue

        obj = json.loads(line)
        entry_id = obj.get("id", f"entry_{i}")

        md.write(f"# Entry {entry_id}\n\n")

        for key, value in obj.items():
            md.write(f"## {key}\n\n")
            md.write("```text\n")
            md.write(str(value))
            md.write("\n```\n\n")

        md.write("\n---\n\n")

print("Wrote Markdown to:", OUTPUT_MD)
