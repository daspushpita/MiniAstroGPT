import json
import random
from pathlib import Path

INPUT_JSONL = Path(
    "/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/"
    "MiniAstroLM/test_data/teacher_output/train.jsonl"
)

OUTPUT_JSONL = Path("data/inspection_subset.jsonl")
OUTPUT_MD    = Path("data/inspection_subset.md")

RANDOM_SEED = 42
N_RANDOM = 15
N_TOTAL = 25

random.seed(RANDOM_SEED)

# ---------- load all entries ----------
entries = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))

assert len(entries) >= N_TOTAL, "Not enough entries to sample from."

# ---------- helpers ----------
def get_text_len(obj):
    text = obj.get("response") or obj.get("output") or ""
    return len(text)

def get_judge_score(obj):
    return obj.get("judge_score", None)

def was_repaired(obj):
    return bool(obj.get("repair_count", 0)) or bool(obj.get("was_repaired", False))

# ---------- random sample ----------
random_samples = random.sample(entries, N_RANDOM)
picked_ids = {id(x) for x in random_samples}

# ---------- risky samples ----------
remaining = [e for e in entries if id(e) not in picked_ids]

# longest / shortest
remaining_sorted_len = sorted(remaining, key=get_text_len)
shortest = remaining_sorted_len[:3]
longest = remaining_sorted_len[-3:]

# repaired outputs
repaired = [e for e in remaining if was_repaired(e)][:2]

# borderline judge scores
scores = [(e, get_judge_score(e)) for e in remaining if get_judge_score(e) is not None]
scores_sorted = sorted(scores, key=lambda x: abs(x[1] - 0.5))
borderline = [e for e, _ in scores_sorted[:2]]

risky = []
for group in (shortest, longest, repaired, borderline):
    for e in group:
        if id(e) not in picked_ids and len(risky) < (N_TOTAL - N_RANDOM):
            risky.append(e)
            picked_ids.add(id(e))

# ---------- final selection ----------
final_samples = (random_samples + risky)[:N_TOTAL]

# ---------- write JSONL ----------
OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for obj in final_samples:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------- write Markdown ----------
with open(OUTPUT_MD, "w", encoding="utf-8") as md:
    md.write("# Human Inspection Subset\n\n")
    md.write(
        "Auto-selected samples for manual review "
        f"(random + risk-based, seed={RANDOM_SEED}).\n\n"
    )
    md.write("---\n\n")

    for i, obj in enumerate(final_samples, start=1):
        entry_id = obj.get("id", f"entry_{i}")
        md.write(f"## Entry {i}: {entry_id}\n\n")

        for key, value in obj.items():
            md.write(f"### {key}\n\n")
            md.write("```text\n")
            md.write(str(value))
            md.write("\n```\n\n")

        md.write("---\n\n")

print(f"Selected {len(final_samples)} samples")
print(f"→ JSONL: {OUTPUT_JSONL}")
print(f"→ Markdown: {OUTPUT_MD}")