from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a held-out teacher eval split.")
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-ids", type=Path, required=True)
    parser.add_argument("--size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_ids.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.database_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title_clean, abstract_clean
            FROM papers
            WHERE abstract_clean IS NOT NULL
            ORDER BY id
            LIMIT ?
            """,
            (args.size,),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    with args.output_jsonl.open("w", encoding="utf-8") as f_jsonl, args.output_ids.open("w", encoding="utf-8") as f_ids:
        for paper_id, title, abstract in rows:
            payload = {
                "id": paper_id,
                "title": title,
                "input": abstract,
            }
            f_jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f_ids.write(f"{paper_id}\n")

    print(f"Wrote {len(rows)} held-out eval rows to {args.output_jsonl}")
    print(f"Wrote held-out IDs to {args.output_ids}")


if __name__ == "__main__":
    main()
