"""Feasibility test for MiniCheck: score LLM-AggreFact with a released checker.

Mirrors the authors' benchmark_evaluation_demo.ipynb (balanced accuracy per dataset,
threshold 0.5) and additionally records runtime and saves raw probabilities.

Usage:
    python minicheck_eval.py <model_name> <out_dir> [--limit N]
    model_name: roberta-large | deberta-v3-large | flan-t5-large
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
from minicheck.minicheck import MiniCheck

model_name, out_dir = sys.argv[1], Path(sys.argv[2])
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(load_dataset("lytang/LLM-AggreFact")["test"])
# The benchmark has grown since the paper; keep only the paper's 10 datasets (Table 2)
PAPER_TABLE2 = {  # BAcc reported in Tang et al. (EMNLP 2024), Table 2
    "roberta-large":    [63.7, 70.8, 71.9, 75.9, 67.6, 88.8, 77.4, 73.3, 57.4, 84.4],
    "deberta-v3-large": [64.2, 71.0, 69.3, 72.7, 69.4, 87.3, 75.6, 73.0, 58.9, 83.9],
    "flan-t5-large":    [69.9, 74.3, 73.6, 77.3, 72.2, 86.2, 74.6, 74.7, 59.0, 85.2],
}
PAPER_DATASETS = ["AggreFact-CNN", "AggreFact-XSum", "TofuEval-MediaS", "TofuEval-MeetB", "Wice",
                  "Reveal", "ClaimVerify", "FactCheck-GPT", "ExpertQA", "Lfqa"]
if "--all" not in sys.argv:
    df = df[df.dataset.isin(PAPER_DATASETS)].reset_index(drop=True)
if limit:  # stratified small sample per dataset for a quick smoke test
    df = df.groupby("dataset").head(limit).reset_index(drop=True)
print(f"{len(df)} claims, {df.dataset.nunique()} datasets", flush=True)

t0 = time.time()
scorer = MiniCheck(model_name=model_name, cache_dir=str(out_dir.parent / "ckpts"))
t_load = time.time() - t0

t0 = time.time()
pred_label, raw_prob, _, _ = scorer.score(docs=list(df.doc.values), claims=list(df.claim.values))
t_pred = time.time() - t0

df["pred"] = pred_label
df["prob"] = raw_prob
rows = []
for name, sub in df.groupby("dataset", sort=False):
    rows.append({"dataset": name, "n": len(sub), "bacc": 100 * balanced_accuracy_score(sub.label, sub.pred)})
res = pd.DataFrame(rows)
paper = dict(zip(PAPER_DATASETS, PAPER_TABLE2.get(model_name, [None] * 10)))
res["paper"] = res.dataset.map(paper)
res["diff"] = res.bacc - res.paper
summary = {
    "model": model_name,
    "n_claims": len(df),
    "load_seconds": round(t_load, 1),
    "predict_seconds": round(t_pred, 1),
    "per_dataset": res.round(2).to_dict(orient="records"),
    "average_all": round(res.bacc.mean(), 2),
    "paper_average": round(sum(PAPER_TABLE2[model_name]) / 10, 2) if model_name in PAPER_TABLE2 else None,
}
df[["dataset", "label", "pred", "prob"]].to_csv(out_dir / f"{model_name}_preds.csv", index=False)
(out_dir / f"{model_name}_summary.json").write_text(json.dumps(summary, indent=2))
print(res.round(1).to_string(index=False))
print(json.dumps({k: v for k, v in summary.items() if k != "per_dataset"}, indent=2))
