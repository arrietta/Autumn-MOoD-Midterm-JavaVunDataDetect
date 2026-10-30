import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import LABEL_OF_STATUS, ML_READY, RAW, SPLITS, VERSIONS, WAREHOUSE, LOGS

REQUIRED = ["code", "idx", "cwe", "target", "status", "commit", "function_name", "body_hash"]

SCHEMA = """
DROP TABLE IF EXISTS fact_sample; DROP TABLE IF EXISTS dim_function; DROP TABLE IF EXISTS dim_commit;
DROP TABLE IF EXISTS dim_cwe; DROP TABLE IF EXISTS dim_dataset; DROP TABLE IF EXISTS etl_run;
CREATE TABLE dim_dataset (dataset_version TEXT PRIMARY KEY, description TEXT, source TEXT);
CREATE TABLE dim_commit  (commit_id TEXT PRIMARY KEY);
CREATE TABLE dim_cwe     (cwe_id TEXT PRIMARY KEY, cwe_kind TEXT);
CREATE TABLE dim_function(function_id TEXT PRIMARY KEY, function_name TEXT, code TEXT,
                          n_chars INTEGER, n_lines INTEGER, norm_hash TEXT);
CREATE TABLE fact_sample (sample_id TEXT PRIMARY KEY, dataset_version TEXT REFERENCES dim_dataset,
                          split TEXT, function_id TEXT REFERENCES dim_function,
                          commit_id TEXT REFERENCES dim_commit, cwe_id TEXT REFERENCES dim_cwe,
                          status TEXT, label INTEGER CHECK (label IN (0,1)));
CREATE TABLE etl_run     (run_at TEXT, step TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS experiment (run_id TEXT PRIMARY KEY, model TEXT, dataset_version TEXT,
                          config TEXT, started_at TEXT, finished_at TEXT, device TEXT);
CREATE TABLE IF NOT EXISTS metric (run_id TEXT REFERENCES experiment, split TEXT, metric TEXT, value REAL);
CREATE TABLE IF NOT EXISTS prediction (run_id TEXT REFERENCES experiment, sample_id TEXT REFERENCES fact_sample,
                          prob REAL, pred INTEGER);
CREATE INDEX IF NOT EXISTS ix_fact_version ON fact_sample(dataset_version, split);
"""


def norm_hash(code):
    return hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()[:16]


def cwe_kind(c):
    if re.fullmatch(r"CWE-\d+", c):
        return "cwe"
    return "nvd_placeholder" if c.startswith("NVD-CWE") else "invalid"


def extract():
    frames = []
    for v in VERSIONS:
        for raw_split, split in SPLITS.items():
            d = pd.read_json(RAW / v / f"{raw_split}.jsonl", lines=True, dtype={"idx": str, "target": str})
            frames.append(d.assign(dataset_version=v, split=split))
    return pd.concat(frames, ignore_index=True)


def transform(df, log):
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    assert not missing_cols, f"schema error, missing {missing_cols}"
    n0 = len(df)
    df["code"] = df.code.astype(str)
    df["cwe"] = df.cwe.astype(str).str.strip().str.upper().str.replace("_", "-").replace(
        {"NVD-CWE-NOINFO": "NVD-CWE-noinfo", "NVD-CWE-OTHER": "NVD-CWE-Other"})
    df["label"] = pd.to_numeric(df.target, errors="coerce")
    bad_label = ~df.label.isin([0, 1])
    inconsistent = df.status.map(LABEL_OF_STATUS) != df.label
    empty = df.code.str.strip() == ""
    drop = bad_label | inconsistent | empty
    log("transform", f"invalid label {int(bad_label.sum())}, status/label mismatch {int(inconsistent.sum())}, "
                     f"empty code {int(empty.sum())} -> dropped {int(drop.sum())}")
    df = df[~drop].copy()
    df["label"] = df.label.astype(int)
    df["norm_hash"] = df.code.map(norm_hash)
    conflict = df.groupby(["dataset_version", "norm_hash"]).label.transform("nunique") > 1
    log("transform", f"same code with conflicting labels dropped {int(conflict.sum())}")
    df = df[~conflict].copy()
    cross = df.groupby(["dataset_version", "norm_hash"]).split.transform("nunique") > 1
    log("transform", f"rows whose code also appears in another split (leakage) {int(cross.sum())}")
    order = df.split.map({"train": 0, "val": 1, "test": 2})
    df = df.assign(_o=order).sort_values(["dataset_version", "_o"], kind="stable")
    dup = df.duplicated(["dataset_version", "norm_hash"])
    log("transform", f"normalized-code duplicates removed {int(dup.sum())}")
    df = df[~dup].drop(columns="_o")
    df["n_chars"] = df.code.str.len()
    df["n_lines"] = df.code.str.count("\n") + 1
    df["sample_id"] = df.dataset_version + ":" + df.split + ":" + df.idx.astype(str)
    log("transform", f"{n0} -> {len(df)} rows")
    return df


def load(df, log_rows):
    WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(WAREHOUSE)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO dim_dataset VALUES (?,?,?)", [
        ("with_p3", "X1 with P3: fix-commit functions + unchanged functions of the same files (~1:32)",
         "github.com/rmusab/vul-llm-finetune"),
        ("without_p3", "X1 without P3: vulnerable/fixed function pairs only (~1:1)",
         "github.com/rmusab/vul-llm-finetune")])
    pd.DataFrame({"commit_id": df.commit.unique()}).to_sql("dim_commit", con, if_exists="append", index=False)
    cw = pd.DataFrame({"cwe_id": df.cwe.unique()})
    cw.assign(cwe_kind=cw.cwe_id.map(cwe_kind)).to_sql("dim_cwe", con, if_exists="append", index=False)
    fn = df.drop_duplicates("body_hash")[["body_hash", "function_name", "code", "n_chars", "n_lines", "norm_hash"]]
    fn.rename(columns={"body_hash": "function_id"}).to_sql("dim_function", con, if_exists="append", index=False)
    df[["sample_id", "dataset_version", "split", "body_hash", "commit", "cwe", "status", "label"]].rename(
        columns={"body_hash": "function_id", "commit": "commit_id", "cwe": "cwe_id"}).to_sql(
        "fact_sample", con, if_exists="append", index=False)
    pd.DataFrame(log_rows, columns=["run_at", "step", "detail"]).to_sql("etl_run", con, if_exists="append", index=False)
    con.commit()
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["dim_dataset", "dim_commit", "dim_cwe", "dim_function", "fact_sample"]}
    con.close()

    ML_READY.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "dataset_version", "split", "commit", "cwe", "function_name", "n_chars", "label", "code"]
    for (v, s), g in df.groupby(["dataset_version", "split"]):
        g[cols].to_csv(ML_READY / f"{v}_{s}.csv", index=False)
    return counts


def main():
    rows = []
    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    log = lambda step, detail: (rows.append((now(), step, detail)), print(f"[{step}] {detail}"))
    raw = extract()
    log("extract", f"{len(raw)} rows from {len(VERSIONS) * 3} jsonl files")
    df = transform(raw, log)
    counts = load(df, rows)
    log("load", json.dumps(counts))
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "etl_log.txt").write_text("\n".join(f"{a}  {b:9s} {c}" for a, b, c in rows) + "\n")


if __name__ == "__main__":
    main()
