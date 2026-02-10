from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]      # project root (one level above tests/)
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from miniastrolm.llm.config import TeacherConfig
from miniastrolm.llm.llamacpp_teacher import LlamaCppTeacher
from miniastrolm.eval.judge import (
    JudgePromptBuilder,
    JudgeValidator,
    JudgeRepairPromptBuilder_v2,
    JudgeReevalPromptBuilder,
    LLMJudge
)
from miniastrolm.eval.judge_config import JudgeConfig


OUT_PATH = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/evals/judge_results_v1.jsonl")
OUT_MD   = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/evals/judge_results_v1.md")

IN_PATH  = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/teacher_output/v2/train.jsonl")
PROMPT_PATH = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/prompts/judge/prompt_v2.txt")
N = 1000

judge_config = JudgeConfig(
    min_faithfulness_to_keep=4,
    min_overall_to_keep=60,
)

llm_config = TeacherConfig(
    n_ctx=4096,         # Perfect for abstracts + 800 word essays
    n_batch=512,        # Higher batch = faster prompt processing
    n_gpu_layers=-1,    # Offload everything
    n_threads=4,        # Use only the 4 "Performance" cores
    f16_kv=True,        # Ensure memory is saved as FP16
)

def load_n(path: Path, n: int):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def write_md_block(f, r: dict):
    j = r.get("judge") or {}
    scores = j.get("scores") or {}
    tags = j.get("error_tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]

    f.write(f"## Paper: {r.get('id','')}\n")
    f.write(f"**Accepted:** {'✅' if r.get('accepted') else '❌'}  \n")
    f.write(f"**Faithfulness:** {scores.get('faithfulness')}  \n")
    f.write(f"**Overall:** {scores.get('overall')}  \n")
    f.write(f"**Error tags:** {', '.join(tags)}\n\n")
    f.write(f"**Rationale:**  \n{j.get('rationale','')}\n\n")
    f.write(f"**Rewrite hint:**  \n{j.get('rewrite_hint','')}\n\n")
    f.write("---\n\n")


my_teacher = LlamaCppTeacher(my_config=llm_config,
                             enable_grammer=0)

# IMPORTANT: your JudgeConfig faithfulness is 0-5 (int), not 0-1 float
judge = LLMJudge(
    llm_client=my_teacher,
    prompt_path=PROMPT_PATH,
    prompt_builder = JudgePromptBuilder(prompt_path=PROMPT_PATH),
    judge_config=judge_config,llm_config=llm_config,
    validator=JudgeValidator(),
    repair_prompt_builder=JudgeRepairPromptBuilder_v2(),
    re_eval_prompt_builder=JudgeReevalPromptBuilder(),
    max_attempts=3,
)

items = load_n(IN_PATH, N)

# Stream output continuously (append + flush)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PATH, "a", encoding="utf-8") as f_jsonl, open(OUT_MD, "a", encoding="utf-8") as f_md:
    for i, item in enumerate(items):
        paper_id = item.get("id") or item.get("paper_id")
        abstract = item.get("input") or item.get("abstract")
        explanation = item.get("output") or item.get("explanation")

        # guard against malformed rows
        if not paper_id or not abstract or explanation is None:
            result = {
                "id": paper_id or "MISSING_ID",
                "judge": None,
                "accepted": False,
                "errors": ["missing_fields_in_input_item"],
                "last_raw": "",
                "metrics": {},
            }
        else:
            result = judge.judge_one(
                paper_id=paper_id,
                abstract=abstract,
                explanation=explanation,
            )

        # write JSONL immediately
        f_jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
        f_jsonl.flush()

        # write MD immediately
        write_md_block(f_md, result)
        f_md.flush()

        print(
            f"[{i}] id={paper_id} "
            f"accepted={result.get('accepted')} "
            f"errors={result.get('errors')}"
        )

print(f"Saved streamed results to:\n- {OUT_PATH}\n- {OUT_MD}")