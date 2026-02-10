from pathlib import Path
import sqlite3

DB = Path("/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/processed/mini_astrolm.db")

def q(cur, sql):
    cur.execute(sql)
    return cur.fetchall()

def main():
    assert DB.exists(), f"DB not found: {DB}"
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    print("DB:", DB)

    # tables
    tables = q(cur, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    print("Tables:", [t[0] for t in tables])

    # schema
    print("\nSchema: papers")
    for row in q(cur, "PRAGMA table_info(papers);"):
        # (cid, name, type, notnull, dflt_value, pk)
        print(row)

    # counts
    n = q(cur, "SELECT COUNT(*) FROM papers;")[0][0]
    print("\nRows in papers:", n)

    nulls = q(cur, """
        SELECT
          SUM(abstract_clean IS NULL),
          SUM(abstract_clean = ''),
          SUM(title_clean IS NULL),
          SUM(title_clean = '')
        FROM papers;
    """)[0]
    print("Null/empty counts (abstract_clean NULL, abstract_clean empty, title_clean NULL, title_clean empty):", nulls)

    lens = q(cur, """
        SELECT
          MIN(LENGTH(abstract_clean)),
          AVG(LENGTH(abstract_clean)),
          MAX(LENGTH(abstract_clean))
        FROM papers
        WHERE abstract_clean IS NOT NULL;
    """)[0]
    print("abstract_clean length (min, avg, max):", lens)

    # year sanity
    year_stats = q(cur, "SELECT MIN(year), MAX(year), COUNT(*) FROM papers WHERE year IS NOT NULL;")[0]
    print("year (min, max, non-null count):", year_stats)

    # random samples
    print("\nSample rows:")
    for (pid, year, title, abs_) in q(cur, """
        SELECT id, year,
               SUBSTR(title_clean, 1, 90),
               SUBSTR(abstract_clean, 1, 160)
        FROM papers
        ORDER BY RANDOM()
        LIMIT 5;
    """):
        print("\n-", pid, "|", year)
        print("  title:", title)
        print("  abstract:", abs_)

    conn.close()

if __name__ == "__main__":
    main()
