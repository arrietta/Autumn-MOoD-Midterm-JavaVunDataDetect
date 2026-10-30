import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS
from metrics_utils import finalize, load_split, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["with_p3", "without_p3"], default="with_p3")
    args = ap.parse_args()
    tr, va, te = (load_split(args.data, s) for s in ("train", "val", "test"))
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", ngram_range=(1, 2), min_df=2,
                          max_features=100000, sublinear_tf=True, lowercase=False)
    Xtr, Xva, Xte = vec.fit_transform(tr.code), vec.transform(va.code), vec.transform(te.code)
    best = None
    for C in (0.3, 1, 3, 10):
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, tr.label)
        pv = m.predict_proba(Xva)[:, 1]
        ap_ = metrics(va.label.to_numpy(), pv, 0.5)["pr_auc"]
        if best is None or ap_ > best[0]:
            best = (ap_, C, m, pv)
    _, C, m, pv = best
    pt = m.predict_proba(Xte)[:, 1]
    res, _ = finalize(RESULTS / "runs" / f"tfidf_lr_{args.data}", va, te, pv, pt, {"C": C})
    print("tfidf", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res["test"].items()}))

    lv, lt = va.n_chars.to_numpy(float), te.n_chars.to_numpy(float)
    res, _ = finalize(RESULTS / "runs" / f"length_only_{args.data}", va, te, lv / lt.max(), lt / lt.max())
    print("length", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res["test"].items()}))


if __name__ == "__main__":
    main()
