import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES, RESULTS

NAMES = {"codebert_lora": "CodeBERT + LoRA", "distilbert_head": "DistilBERT + head", "tfidf_lr": "TF-IDF + LogReg", "length_only": "Length only"}
COLORS = {"codebert_lora": "#1B5E9E", "distilbert_head": "#D9822B", "tfidf_lr": "#7A869A", "length_only": "#B8C2CC"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def runs():
    out = []
    for d in sorted((RESULTS / "runs").glob("*")):
        if not (d / "metrics.json").exists() or not (d / "test_predictions.csv").exists():
            continue
        model = next(k for k in NAMES if d.name.startswith(k))
        data = d.name[len(model) + 1:].split("_tag")[0]
        if data not in ("with_p3", "without_p3"):
            continue
        out.append({"run": d.name, "model": model, "data": data, "dir": d,
                    "m": json.loads((d / "metrics.json").read_text()),
                    "pred": pd.read_csv(d / "test_predictions.csv")})
    return out


def table(rs):
    rows = []
    for r in rs:
        t = r["m"]["test"]
        cfg = json.loads((r["dir"] / "config.json").read_text()) if (r["dir"] / "config.json").exists() else {}
        rows.append({"dataset": r["data"], "model": NAMES[r["model"]], "accuracy": t["accuracy"],
                     "majority_baseline": t["majority_baseline_accuracy"], "balanced_acc": t["balanced_accuracy"],
                     "precision": t["precision"], "recall": t["recall"], "f1": t["f1"], "roc_auc": t["roc_auc"],
                     "pr_auc": t["pr_auc"], "mcc": t["mcc"], "tp": t["tp"], "fp": t["fp"], "fn": t["fn"], "tn": t["tn"],
                     "threshold": t["threshold"], "trainable_params": cfg.get("trainable_params"),
                     **{f"triage_{k}": r["m"]["test_triage"][k] for k in ("threshold", "accuracy", "precision", "recall", "f1", "mcc", "tp", "fp")
                        if "test_triage" in r["m"]},
                     "train_minutes": r["m"].get("train_minutes")})
    t = pd.DataFrame(rows).sort_values(["dataset", "pr_auc"], ascending=[True, False])
    t.to_csv(RESULTS / "model_comparison.csv", index=False)
    return t


def curves(rs, data):
    sel = [r for r in rs if r["data"] == data]
    if not sel:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for r in sel:
        y, p = r["pred"].label, r["pred"].prob
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        c = COLORS[r["model"]]
        ax[0].plot(fpr, tpr, color=c, lw=1.6, label=f"{NAMES[r['model']]}  AUC {r['m']['test']['roc_auc']:.3f}")
        ax[1].plot(rc, pr, color=c, lw=1.6, label=f"{NAMES[r['model']]}  AP {r['m']['test']['pr_auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], ls="--", color="#BBB", lw=1)
    ax[1].axhline(sel[0]["pred"].label.mean(), ls="--", color="#BBB", lw=1, label="random (prevalence)")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curve, test ({data})")
    ax[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall curve, test ({data})")
    for a in ax:
        a.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"eval_curves_{data}.png", dpi=170)
    plt.close(fig)


def confusion(rs, data):
    sel = [r for r in rs if r["data"] == data]
    if not sel:
        return
    fig, axes = plt.subplots(1, len(sel), figsize=(3.6 * len(sel), 3.3))
    for ax, r in zip(np.atleast_1d(axes), sel):
        t = r["m"]["test"]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]}\n({share:.0%})", ha="center", va="center", fontsize=10,
                        color="white" if share > 0.6 else "#1F2933")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{NAMES[r['model']]}\nthreshold {t['threshold']:.2f}", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"eval_confusion_{data}.png", dpi=170)
    plt.close(fig)


def loss_curves(rs):
    sel = [r for r in rs if (r["dir"] / "loss_history.csv").exists()]
    if not sel:
        return
    fig, ax = plt.subplots(1, len(sel), figsize=(4.2 * len(sel), 3.2), squeeze=False)
    for a, r in zip(ax[0], sel):
        h = pd.read_csv(r["dir"] / "loss_history.csv")
        a.plot(h.step, h.loss, color="#C8D1DC", lw=0.6)
        sm = h.loss.rolling(50, min_periods=1).mean()
        a.plot(h.step, sm, color=COLORS[r["model"]], lw=1.6, label="train loss (avg 50)")
        a.set_ylim(0, sm.max() * 1.6)
        for _, e in h.dropna(subset=["val_pr_auc"]).iterrows():
            a.axvline(e.step, color="#DDD", lw=0.8)
            a.text(e.step, a.get_ylim()[1] * 0.92, f"val AP {e.val_pr_auc:.2f}", fontsize=7, ha="right", color="#52606D")
        a.set(title=f"{NAMES[r['model']]} ({r['data']})", xlabel="step", ylabel="weighted CE loss")
    fig.tight_layout()
    fig.savefig(FIGURES / "eval_training_loss.png", dpi=170)
    plt.close(fig)


def per_cwe(rs):
    sel = [r for r in rs if r["data"] == "with_p3"]
    if not sel:
        return None
    frames = []
    for r in sel:
        p = r["pred"][r["pred"].label == 1]
        frames.append(p.groupby("cwe").pred.agg(["size", "mean"]).rename(columns={"mean": NAMES[r["model"]]}))
    t = pd.concat(frames, axis=1)
    t = t.loc[:, ~t.columns.duplicated()].rename(columns={"size": "test_positives"})
    t = t.sort_values("test_positives", ascending=False)
    t.to_csv(RESULTS / "per_cwe_recall.csv")
    return t


def topk(rs):
    rows = []
    for r in rs:
        if r["data"] != "with_p3":
            continue
        p = r["pred"].sort_values("prob", ascending=False)
        for k in (1, 2, 5, 10, 20):
            n = int(round(len(p) * k / 100))
            found = int(p.head(n).label.sum())
            rows.append({"model": NAMES[r["model"]], "review_top_%": k, "functions_reviewed": n, "vulns_found": found,
                         "share_of_all_vulns_%": round(100 * found / p.label.sum(), 1)})
    t = pd.DataFrame(rows)
    if len(t):
        t.pivot(index="review_top_%", columns="model", values="share_of_all_vulns_%").to_csv(RESULTS / "topk_review_all_models.csv")
        t[t.model == NAMES["codebert_lora"]].drop(columns="model").to_csv(RESULTS / "topk_review.csv", index=False)
    return t


def topk_plot(rs):
    sel = [r for r in rs if r["data"] == "with_p3"]
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.linspace(0, 30, 301)
    for r in sorted(sel, key=lambda r: -r["m"]["test"]["pr_auc"]):
        p = r["pred"].sort_values("prob", ascending=False).label.to_numpy()
        found = np.array([p[:int(round(len(p) * k / 100))].sum() for k in x]) / p.sum() * 100
        ax.plot(x, found, color=COLORS[r["model"]], lw=2 if r["model"] == "codebert_lora" else 1.3, label=NAMES[r["model"]])
    ax.plot(x, x, ls="--", color="#BBB", lw=1, label="random order")
    ax.axvline(5, color="#DDD", lw=0.8)
    ax.set(xlabel="% of functions reviewed (highest risk first)", ylabel="% of all vulnerabilities found",
           title="How much of the review effort the ranking saves", xlim=(0, 30), ylim=(0, 100))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "eval_topk.png", dpi=170)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rs = runs()
    t = table(rs)
    for d in ("with_p3", "without_p3"):
        curves(rs, d)
        confusion(rs, d)
    loss_curves(rs)
    per_cwe(rs)
    topk(rs)
    topk_plot(rs)
    pd.set_option("display.width", 200)
    print(t.drop(columns=["tp", "fp", "fn", "tn"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
