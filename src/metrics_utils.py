import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).parent))
from config import ML_READY, SEED


def load_split(version, split, limit=None):
    d = pd.read_csv(ML_READY / f"{version}_{split}.csv", keep_default_na=False)
    if limit:
        pos = d[d.label == 1].head(max(2, limit // 10))
        d = pd.concat([pos, d[d.label == 0].sample(min(limit - len(pos), (d.label == 0).sum()), random_state=SEED)])
    return d.reset_index(drop=True)


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "mcc": matthews_corrcoef(y, pred), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean()))}


def threshold_triage(y, p):
    cand = np.unique(np.quantile(p, np.linspace(0.0, 0.999, 500)))
    return float(max(cand, key=lambda t: matthews_corrcoef(y, p >= t)))


MIN_ACCURACY = 0.93


def threshold_alert(y, p):
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 500)))
    ok = [t for t in cand if accuracy_score(y, p >= t) >= MIN_ACCURACY]
    return float(max(ok or cand, key=lambda t: f1_score(y, p >= t)))


best_threshold = threshold_triage


def finalize(out, va, te, pv, pt, extra=None):
    import json
    yv, yt = va.label.to_numpy(), te.label.to_numpy()
    ta, tt = threshold_alert(yv, pv), threshold_triage(yv, pv)
    res = dict(extra or {})
    res.update({"val": metrics(yv, pv, ta), "test": metrics(yt, pt, ta),
                "val_triage": metrics(yv, pv, tt), "test_triage": metrics(yt, pt, tt),
                "test_at_0.5": metrics(yt, pt, 0.5)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    cols = ["sample_id", "cwe", "n_chars", "label", "prob", "pred", "pred_triage"]
    va.assign(prob=pv, pred=(pv >= ta).astype(int), pred_triage=(pv >= tt).astype(int))[cols].to_csv(out / "val_predictions.csv", index=False)
    te.assign(prob=pt, pred=(pt >= ta).astype(int), pred_triage=(pt >= tt).astype(int))[cols].to_csv(out / "test_predictions.csv", index=False)
    return res, ta
