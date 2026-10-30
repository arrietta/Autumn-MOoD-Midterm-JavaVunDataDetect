import argparse
import json
import math
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, MODELS_CFG, RESULTS, SEED, WAREHOUSE
from metrics_utils import finalize, load_split, metrics


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def batches(enc, labels, batch_size, shuffle, rng=None):
    lengths = np.array([len(x) for x in enc])
    order = np.argsort(lengths, kind="stable")
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(chunks)
    for idx in chunks:
        m = max(lengths[idx])
        ids = torch.full((len(idx), m), PAD_ID, dtype=torch.long)
        att = torch.zeros((len(idx), m), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :lengths[i]] = torch.tensor(enc[i])
            att[r, :lengths[i]] = 1
        yield idx, ids, att, torch.tensor(labels[idx], dtype=torch.long)


@torch.no_grad()
def predict(model, enc, labels, bs, device):
    model.eval()
    probs = np.zeros(len(enc))
    for idx, ids, att, _ in batches(enc, labels, bs * 2, shuffle=False):
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
        probs[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return probs


def build_model(cfg):
    model = AutoModelForSequenceClassification.from_pretrained(cfg["hf_name"], num_labels=2)
    if cfg["lora"]:
        from peft import LoraConfig, TaskType, get_peft_model
        lc = cfg["lora"]
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=lc["r"], lora_alpha=lc["alpha"],
                                                 lora_dropout=lc["dropout"], target_modules=lc["target_modules"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS_CFG), required=True)
    ap.add_argument("--data", choices=["with_p3", "without_p3"], default="with_p3")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-len", type=int, help="override max tokens (faster runs)")
    ap.add_argument("--limit", type=int, help="subsample every split (smoke test)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-only", action="store_true", help="re-evaluate a saved model (models/<run_id>) without training")
    args = ap.parse_args()

    cfg = dict(MODELS_CFG[args.model])
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.max_len:
        cfg["max_len"] = args.max_len
    set_seed(SEED)
    device = pick_device()
    run_id = f"{args.model}_{args.data}{'_' + args.tag if args.tag else ''}"
    out = RESULTS / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a" if args.eval_only else "w")

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    tr, va, te = (load_split(args.data, s, args.limit) for s in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(cfg["hf_name"])
    global PAD_ID
    PAD_ID = tok.pad_token_id
    enc = {n: tok(list(d.code), truncation=True, max_length=cfg["max_len"])["input_ids"] for n, d in
           (("train", tr), ("val", va), ("test", te))}
    y = {n: d.label.to_numpy() for n, d in (("train", tr), ("val", va), ("test", te))}
    trunc = np.mean([len(x) >= cfg["max_len"] for x in enc["train"]])

    if args.eval_only:
        return eval_only(run_id, cfg, out, tok, enc, y, va, te, device, log)
    model, trainable, total = build_model(cfg)
    model.to(device)
    pos_rate = y["train"].mean()
    w_pos = float(math.sqrt((1 - pos_rate) / pos_rate))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, w_pos], device=device))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.01)
    steps = cfg["epochs"] * math.ceil(len(enc["train"]) / cfg["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    config = {"run_id": run_id, "model": args.model, "data": args.data, "device": device, "seed": SEED, **cfg,
              "train_rows": len(tr), "val_rows": len(va), "test_rows": len(te), "train_pos_rate": float(pos_rate),
              "class_weight_pos": w_pos, "trainable_params": trainable, "total_params": total,
              "truncated_train_%": round(100 * trunc, 2), "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log(f"run {run_id} on {device}: train {len(tr)} (pos {pos_rate:.2%}), val {len(va)}, test {len(te)}")
    log(f"params trainable {trainable:,} / total {total:,} ({100 * trainable / total:.2f}%), pos weight {w_pos:.2f}, "
        f"truncated at {cfg['max_len']} tokens: {100 * trunc:.1f}% of train")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rng = np.random.default_rng(SEED)
    hist, best, step, t0 = [], None, 0, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for _, ids, att, lab in batches(enc["train"], y["train"], cfg["batch_size"], True, rng):
            logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
            loss = loss_fn(logits.float(), lab.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            hist.append({"epoch": ep, "step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
            if step % 50 == 0:
                recent = np.mean([h["loss"] for h in hist[-50:]])
                log(f"epoch {ep} step {step}/{steps} loss {recent:.4f} ({(time.time() - t0) / step:.2f}s/step)")
        pv = predict(model, enc["val"], y["val"], cfg["batch_size"], device)
        val_auc, val_ap = roc_auc_score(y["val"], pv), average_precision_score(y["val"], pv)
        log(f"epoch {ep} done: val ROC-AUC {val_auc:.4f}  PR-AUC {val_ap:.4f}")
        hist[-1].update({"val_roc_auc": val_auc, "val_pr_auc": val_ap})
        if best is None or val_ap > best[0]:
            best = (val_ap, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, pv)
    pd.DataFrame(hist).to_csv(out / "loss_history.csv", index=False)

    _, best_ep, state, pv = best
    model.load_state_dict(state)
    pt = predict(model, enc["test"], y["test"], cfg["batch_size"], device)
    res, thr = finalize(out, va, te, pv, pt, {"best_epoch": best_ep, "train_minutes": round((time.time() - t0) / 60, 1)})
    log("TEST " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res["test"].items()}))

    if not args.limit:
        save = MODELS / run_id
        model.save_pretrained(save)
        tok.save_pretrained(save)
        con = sqlite3.connect(WAREHOUSE)
        con.execute("DELETE FROM metric WHERE run_id=?", (run_id,))
        con.execute("DELETE FROM prediction WHERE run_id=?", (run_id,))
        con.execute("INSERT OR REPLACE INTO experiment VALUES (?,?,?,?,?,?,?)",
                    (run_id, args.model, args.data, json.dumps(config), started,
                     datetime.now(timezone.utc).isoformat(timespec="seconds"), device))
        con.executemany("INSERT INTO metric VALUES (?,?,?,?)",
                        [(run_id, s, k, float(v)) for s in ("val", "test") for k, v in res[s].items()])
        con.executemany("INSERT INTO prediction VALUES (?,?,?,?)",
                        list(zip([run_id] * len(te), te.sample_id, pt.astype(float), (pt >= thr).astype(int).tolist())))
        con.commit()
        con.close()
    logf.close()


def eval_only(run_id, cfg, out, tok, enc, y, va, te, device, log):
    base = AutoModelForSequenceClassification.from_pretrained(cfg["hf_name"], num_labels=2)
    if cfg["lora"]:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, MODELS / run_id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODELS / run_id, num_labels=2)
    model.to(device)
    old = json.loads((out / "metrics.json").read_text())
    pv = predict(model, enc["val"], y["val"], cfg["batch_size"], device)
    pt = predict(model, enc["test"], y["test"], cfg["batch_size"], device)
    res, _ = finalize(out, va, te, pv, pt, {k: old[k] for k in ("best_epoch", "train_minutes") if k in old})
    log("RE-EVAL TEST alert " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res["test"].items()}))
    log("RE-EVAL TEST triage " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res["test_triage"].items()}))


PAD_ID = 0
if __name__ == "__main__":
    main()
