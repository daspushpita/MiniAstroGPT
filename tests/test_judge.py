from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"

sys.path.insert(0, str(SRC))
from miniastrolm.llm.teacher import Llama_Teacher, Validation_Regeneration
from miniastrolm.eval.judge import JudgePromptBuilder, JudgeValidator, JudgeRepairPromptBuilder, LLMJudge, JudgeResult, JudgeConfig

OUT_PATH = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/test_data/judge_results.jsonl")
all_results = []

def load_n(path, n):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


my_teacher = Llama_Teacher()

judge = LLMJudge(
    llm_client=my_teacher,
    config=JudgeConfig(min_faithfulness_to_keep=0.7),
    prompt_builder=JudgePromptBuilder(),
    validator=JudgeValidator(model_name=my_teacher.model_id),
    repair_prompt_builder=JudgeRepairPromptBuilder(),
    max_attempts=3,
)

items = load_n("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/test_data/teacher_output/train.jsonl", 10)

for i, item in enumerate(items):
    paper_id = item.get("id") or item.get("paper_id")
    abstract = item["input"]
    explanation = item.get("output")

    result = judge.judge_one(
        paper_id=paper_id,
        abstract=abstract,
        explanation=explanation,
    )
    all_results.append(result)


    print(
        f"[{i}] id={paper_id} "
        f"accepted={result['accepted']} "
        f"errors={result['errors']}"
    )
    

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in all_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved {len(all_results)} results to {OUT_PATH}")