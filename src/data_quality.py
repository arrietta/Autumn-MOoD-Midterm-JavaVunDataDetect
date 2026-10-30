import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import etl
from config import FIGURES, RESULTS, WAREHOUSE


def dimensions(df):
    nh = df.code.map(etl.norm_hash)
    lab = pd.to_numeric(df.target if "target" in df else df.label, errors="coerce")
    status = df.status.map(etl.LABEL_OF_STATUS)
    return {
        "rows": int(len(df)),
        "completeness_%": round(100 * (1 - (df[["code", "cwe", "status", "commit", "function_name"]].astype(str)
                                             .apply(lambda c: c.str.strip() == "").to_numpy().mean())), 3),
        "unique_body_hash_%": round(100 * df.body_hash.nunique() / len(df), 3) if "body_hash" in df else None,
        "unique_normalized_code_%": round(100 * nh.nunique() / len(df), 3),
        "label_validity_%": round(100 * lab.isin([0, 1]).mean(), 3),
        "status_label_consistency_%": round(100 * (status == lab).mean(), 3),
        "cwe_proper_code_%": round(100 * df.cwe.str.fullmatch(r"CWE-\d+").mean(), 2),
        "cwe_nvd_placeholder_%": round(100 * df.cwe.str.startswith("NVD-CWE").mean(), 2),
        "commits_in_more_than_one_split": int((df.groupby("commit").split.nunique() > 1).sum()),
        "code_in_more_than_one_split": int((df.groupby(nh).split.nunique() > 1).sum()),
        "same_code_conflicting_labels": int((lab.groupby(nh).nunique() > 1).sum()),
        "positive_%": round(100 * lab.mean(), 2),
        "neg_per_pos": round((1 - lab.mean()) / lab.mean(), 1),
    }


def main():
    raw = etl.extract()
    con = sqlite3.connect(WAREHOUSE)
    clean = pd.read_sql("""SELECT f.*, d.code, f.commit_id AS 'commit', f.cwe_id AS cwe, d.n_chars, d.n_lines, d.function_name,
                                  d.function_id AS body_hash
                           FROM fact_sample f JOIN dim_function d USING(function_id)""", con)
    con.close()
    report = {"before_etl": {}, "after_etl": {}}
    for v in ["with_p3", "without_p3"]:
        report["before_etl"][v] = dimensions(raw[raw.dataset_version == v])
        report["after_etl"][v] = dimensions(clean[clean.dataset_version == v])
    dist = clean.groupby(["dataset_version", "split"]).label.agg(n="size", pos="sum").reset_index()
    dist["neg"] = dist.n - dist.pos
    dist["pos_%"] = (100 * dist.pos / dist.n).round(2)
    report["class_distribution"] = dist.to_dict("records")
    report["length_chars"] = clean.groupby(["dataset_version", "label"]).n_chars.describe(
        percentiles=[.5, .9]).round(0).reset_index().to_dict("records")
    report["top_cwe_positives"] = clean[(clean.dataset_version == "with_p3") & (clean.label == 1)].cwe.value_counts().head(10).to_dict()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "data_quality.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({(k, v): d for k in ("before_etl", "after_etl") for v, d in report[k].items()}).to_csv(RESULTS / "data_quality_table.csv")
    dist.to_csv(RESULTS / "class_distribution.csv", index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for i, v in enumerate(["with_p3", "without_p3"]):
        d = dist[dist.dataset_version == v].set_index("split").loc[["train", "val", "test"]]
        ax[i].bar(d.index, d.neg, label="not vulnerable (0)", color="#4C78A8")
        ax[i].bar(d.index, d.pos, bottom=d.neg, label="vulnerable (1)", color="#E45756")
        for j, (n, p) in enumerate(zip(d.n, d["pos_%"])):
            ax[i].text(j, n, f"{p:.1f}% pos", ha="center", va="bottom", fontsize=8)
        ax[i].set_title(f"{v}: class distribution")
        ax[i].legend(fontsize=7)
    w = clean[clean.dataset_version == "with_p3"]
    for lab, c in ((0, "#4C78A8"), (1, "#E45756")):
        ax[2].hist(w[w.label == lab].n_chars.clip(upper=6000), bins=60, alpha=.6, density=True, color=c, label=f"label {lab}")
    ax[2].set_title("with_p3: function length (chars, clipped at 6000)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "dq_class_and_length.png", dpi=150)
    plt.close(fig)

    top = clean[clean.label == 1].cwe.value_counts().head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(top.index, top.values, color="#E45756")
    ax.set_title("Most frequent CWE among vulnerable functions")
    fig.tight_layout()
    fig.savefig(FIGURES / "dq_top_cwe.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("before_etl", "after_etl")}, indent=1))


if __name__ == "__main__":
    main()
