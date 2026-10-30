import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES

INK, MUTED = "#1F2933", "#52606D"
FILL = {"data": "#DCEBFA", "proc": "#E3F4E8", "ml": "#FCE8D5", "gov": "#EEE6F8", "store": "#F5F0D8"}


def box(ax, x, y, w, h, title, body="", kind="proc", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec=INK, lw=1))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=fs, weight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=fs - 1.6, color=MUTED, linespacing=1.3)


def arrow(ax, a, b, text="", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.1, color=INK,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", fontsize=7, color=MUTED)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def lifecycle(slide=False):
    fig, ax = canvas(16, 5.0)
    steps = [
        ("1. Acquisition", "6 jsonl files from\nvul-llm-finetune\n(NVD fix commits)", "data", "Data owner:\nresearch team"),
        ("2. Validation", "schema, label in {0,1},\nstatus ↔ label,\nempty code", "proc", "DAMA: Data Quality\nISO 8000 / 25012"),
        ("3. Storage", "SQLite star schema\nwarehouse.db\n(read-only raw copy)", "store", "DAMA: Data Storage\nISO 27001 A.8"),
        ("4. Transformation", "dedup, leakage removal,\nCWE normalization,\ntokenization ≤512", "proc", "DAMA: Integration\n& Interoperability"),
        ("5. Training", "CodeBERT + LoRA\nDistilBERT (alt.)\nweighted CE, seed 42", "ml", "Model owner:\nML engineer"),
        ("6. Evaluation", "Acc, P, R, F1,\nROC-AUC, PR-AUC,\nconfusion matrix", "ml", "DAMA: BI &\nAnalytics"),
        ("7. Retention", "keep code, metrics,\nadapters; delete\ncheckpoints after defense", "gov", "ISO 27001 A.8.10\ninformation deletion"),
    ]
    w, gap = 1.95, 0.22
    for i, (t, b, k, g) in enumerate(steps):
        x = 0.3 + i * (w + gap)
        box(ax, x, 2.75, w, 1.45, t, b, k)
        ax.text(x + w / 2, 2.6, g, ha="center", va="top", fontsize=7, color="#7B5EA7", style="italic")
        if i:
            arrow(ax, (x - gap, 3.45), (x, 3.45))
    arrow(ax, (14.9, 4.25), (4.7, 4.25), rad=0.05)
    ax.text(9.8, 4.6, "feedback: errors found in evaluation go back to validation rules", ha="center", fontsize=7, color=MUTED)
    ax.set_ylim(2.2 if slide else 1.25, 4.75 if slide else 5.0)
    if not slide:
        ax.text(8, 1.55, "Information value chain:  raw commits → validated facts → curated training set → model → metrics → "
            "risk-ranked list of Java functions for security review → decision (what to review / fix first)",
            ha="center", fontsize=8.5, color=INK, weight="bold")
    if not slide:
        ax.text(8, 4.85, "Data lifecycle of the vulnerability-detection dataset (Group 6)", ha="center", fontsize=12, weight="bold")
    fig.savefig(FIGURES / ("lifecycle_slide.png" if slide else "lifecycle.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def entity(ax, x, y, name, fields, kind="data", w=2.9):
    h = 0.42 + 0.27 * len(fields)
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.01,rounding_size=0.05", fc="white", ec=INK, lw=1))
    ax.add_patch(FancyBboxPatch((x, y - 0.42), w, 0.42, boxstyle="round,pad=0.01,rounding_size=0.05", fc=FILL[kind], ec=INK, lw=1))
    ax.text(x + w / 2, y - 0.21, name, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
    for i, f in enumerate(fields):
        ax.text(x + 0.12, y - 0.62 - 0.27 * i, f, va="center", fontsize=7.6, color=INK, family="monospace")
    return (x, y - h, w, h)


def er_model():
    fig, ax = canvas(15, 7.4)
    ax.text(7.5, 7.15, "Conceptual / logical data model (warehouse.db, star schema + experiment tables)",
            ha="center", fontsize=12, weight="bold")
    fact = entity(ax, 5.9, 5.0, "fact_sample", ["PK sample_id", "FK dataset_version", "FK function_id",
                                                "FK commit_id", "FK cwe_id", "   split  (train/val/test)",
                                                "   status", "   label  {0,1}"], "proc", 3.2)
    entity(ax, 0.3, 6.6, "dim_dataset", ["PK dataset_version", "   description", "   source"])
    entity(ax, 0.3, 3.9, "dim_function  (Java function)", ["PK function_id (body_hash)", "   function_name", "   code",
                                                            "   n_chars, n_lines", "   norm_hash"], w=3.3)
    entity(ax, 11.6, 6.6, "dim_commit", ["PK commit_id"])
    entity(ax, 11.6, 4.9, "dim_cwe  (vulnerability type)", ["PK cwe_id", "   cwe_kind (cwe/nvd)"], w=3.1)
    entity(ax, 5.9, 2.2, "experiment", ["PK run_id", "   model, dataset_version", "   config (json)", "   device, timestamps"], "ml", 3.2)
    entity(ax, 0.3, 1.5, "metric", ["FK run_id", "   split, metric, value"], "ml")
    entity(ax, 11.6, 2.2, "prediction", ["FK run_id", "FK sample_id", "   prob, pred"], "ml", 3.1)
    rel = [((3.2, 6.1), (5.9, 4.7), "N : 1"), ((3.6, 3.2), (5.9, 3.9), "N : 1"), ((11.6, 6.3), (9.1, 4.7), "N : 1"),
           ((11.6, 4.4), (9.1, 4.2), "N : 1"), ((5.9, 1.3), (3.2, 1.0), "1 : N"), ((9.1, 1.4), (11.6, 1.4), "1 : N"),
           ((12.9, 2.2), (8.6, 2.85), "N : 1 (sample)")]
    for a, b, t in rel:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.1, t, fontsize=7, color=MUTED, ha="center")
    ax.text(7.5, 0.25, "One Java function can appear in both dataset versions (without_p3 ⊂ with_p3); one commit gives "
            "the vulnerable and the fixed version of a function; label = 1 only for status VULNERABLE.",
            ha="center", fontsize=8, color=INK)
    fig.savefig(FIGURES / "er_model.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = canvas(16, 5.9)
    ax.text(8, 5.65, "System architecture (Group 6: CodeBERT + LoRA, alternative DistilBERT)", ha="center", fontsize=12, weight="bold")
    box(ax, 0.3, 3.9, 2.4, 1.4, "Source", "GitHub: vul-llm-finetune\nDatasets/*.tar.gz\n(jsonl, 24 288 rows)", "data")
    box(ax, 3.2, 3.9, 2.6, 1.4, "ETL  (src/etl.py)", "extract → validate →\ndedup / leakage fix →\nload", "proc")
    box(ax, 6.3, 3.9, 2.6, 1.4, "Warehouse", "SQLite star schema\nfact_sample + 4 dims\n+ experiment tables", "store")
    box(ax, 6.3, 1.6, 2.6, 1.4, "Data quality", "src/data_quality.py\ncompleteness, dupes,\nleakage, class balance", "proc")
    box(ax, 9.4, 3.9, 2.9, 1.4, "Training (src/train.py)", "tokenizer (≤512 tokens)\nCodeBERT + LoRA r=16\nweighted CE, MPS GPU", "ml")
    box(ax, 9.4, 1.6, 2.9, 1.4, "Baselines", "DistilBERT + head (alt.)\nTF-IDF + LogReg", "ml")
    box(ax, 12.8, 3.9, 2.9, 1.4, "Evaluation", "threshold on val (MCC)\nAcc, P, R, F1, ROC/PR-AUC\nconfusion matrix", "ml")
    box(ax, 12.8, 1.6, 2.9, 1.4, "Decision layer", "risk-ranked functions\n→ security review queue\n→ report / dashboard", "gov")
    for a, b in [((2.7, 4.6), (3.2, 4.6)), ((5.8, 4.6), (6.3, 4.6)), ((8.9, 4.6), (9.4, 4.6)),
                 ((12.3, 4.6), (12.8, 4.6)), ((7.6, 3.9), (7.6, 3.0)), ((10.85, 3.9), (10.85, 3.0)),
                 ((14.25, 3.9), (14.25, 3.0)), ((12.3, 2.3), (12.8, 2.3))]:
        arrow(ax, a, b)
    ax.text(0.3, 3.2, "Security controls", fontsize=9, weight="bold", color=INK)
    ax.text(0.3, 2.95, "• raw data read-only, hashes logged\n• least-privilege roles (owner /\n  engineer / analyst)\n"
                       "• no secrets in code, local only\n• model + logs versioned", fontsize=7.6, color=MUTED, va="top")
    ax.set_ylim(1.35, 5.9)
    fig.savefig(FIGURES / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    lifecycle()
    er_model()
    architecture()
    print("ok")
