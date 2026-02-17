from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from miniastrolm.llm.config import TeacherConfig
from miniastrolm.llm.llamacpp_teacher import LlamaCppTeacher
from miniastrolm.llm.validation_regeneration import Teacher_Data_Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher v1 data generation with strict filtering.")
    parser.add_argument("--database-path", type=Path, default=ROOT / "data" / "processed" / "mini_astrolm.db")
    parser.add_argument("--output-path", type=Path, default=ROOT / "data" / "teacher" / "v1")
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=ROOT / "prompts" / "teacher" / "teacher_prompt_distill_v1.txt",
    )
    parser.add_argument("--exclude-ids-path", type=Path, default=ROOT / "data" / "teacher" / "v1" / "eval_v1_ids.txt")
    parser.add_argument("--target-accepted", type=int, required=True)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    my_config = TeacherConfig(
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        f16_kv=True,
    )

    teacher_model = LlamaCppTeacher(my_config=my_config)
    pipeline = Teacher_Data_Pipeline(
        database_path=args.database_path,
        output_path=args.output_path,
        prompt_path=args.prompt_path,
        teacher_model=teacher_model,
        my_config=my_config,
        raise_on_fail=False,
        log_skips=False,
        print_every=args.print_every,
        max_total=args.max_total,
        max_accepted=args.target_accepted,
        exclude_ids_path=args.exclude_ids_path,
    )
    pipeline.run()

    execution_time = time.time() - start_time
    print(f"Executed in: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
