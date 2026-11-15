"""Phase 3 metrics from run_checker.py outputs (CPU only), compared with the paper's Table 2.

Usage:
    python metrics.py results/kaggle-t4 [--legacy results/roberta-large/roberta-large_preds.csv]

Writes to <results>/metrics/:
    per_dataset.csv      one row per model / split / dataset, plus 'Average' (macro, the paper's
                         aggregation) and 'Pooled' (all claims together) rows
    summary.csv          one row per model / split
    comparison_test.md   test-split BAcc of every model vs Table 2
    hardware_check.json  (with --legacy) RoBERTa test predictions: earlier laptop run vs this run

Positive class = supported (label 1), as in LLM-AggreFact. So FP = an unsupported claim predicted
supported (a missed error), FN = a supported claim flagged as unsupported (a false alarm).
"""
import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "configs" / "experiment.json").read_text())
DATASETS = CFG["datasets"]
META_KEYS = ["inference_seconds", "latency_ms_mean", "latency_ms_median", "latency_ms_p95",
             "n_chunks_mean", "peak_gpu_mb_allocated", "peak_gpu_mb_reserved", "n_batch_fallbacks"]


def scores(df):
    y, p = df.label.astype(int), df.pred.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    with warnings.catch_warnings():
        if y.nunique() < 2:  # only in tiny smoke-test samples; sklearn then averages over the class present
            warnings.simplefilter("ignore", UserWarning)
        bacc = balanced_accuracy_score(y, p)
    return {"n": len(df), "n_supported": int((y == 1).sum()), "n_unsupported": int((y == 0).sum()),
            "bacc": 100 * bacc,
            "precision_supported": 100 * pr[1], "recall_supported": 100 * rc[1], "f1_supported": 100 * f1[1],
            "precision_unsupported": 100 * pr[0], "recall_unsupported": 100 * rc[0], "f1_unsupported": 100 * f1[0],
            "macro_f1": 100 * f1.mean(), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def load_run(split_dir):
    """Per-dataset predictions and metadata for one model/split; only finished datasets."""
    preds, metas = {}, {}
    for name in DATASETS:
        if (split_dir / f"{name}.csv").exists():
            preds[name] = pd.read_csv(split_dir / f"{name}.csv")
            metas[name] = json.loads((split_dir / f"{name}.json").read_text())
    return preds, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--legacy", type=Path, help="laptop RoBERTa preds CSV (from minicheck_eval.py) to compare against")
    args = ap.parse_args()
    out = args.results / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    rows, summary, test_bacc = [], [], {}
    for model in CFG["models"]:
        paper = CFG["paper_table2"][model]
        paper_ds = dict(zip(DATASETS, paper["bacc"]))
        runs_file = args.results / model / "runs.jsonl"
        runs = [json.loads(l) for l in runs_file.read_text().splitlines()] if runs_file.exists() else []
        full_runs = [r for r in runs if r.get("limit") is None] or runs  # smoke-test folders only have limited runs
        for split in ["dev", "test"]:
            preds, metas = load_run(args.results / model / split)
            if not preds:
                continue
            per = []
            for name, df in preds.items():
                r = {"model": model, "split": split, "dataset": name, **scores(df),
                     **{k: metas[name].get(k) for k in META_KEYS}, "gpu": metas[name]["env"]["gpu"]}
                # Largest BAcc change one flipped prediction can cause on this dataset
                smaller = min(r["n_supported"], r["n_unsupported"])
                r["one_flip_max"] = 50 / smaller if smaller else None
                if split == "test":
                    r["paper_bacc"] = paper_ds[name]
                    r["diff"] = r["bacc"] - paper_ds[name]
                    r["matches_paper"] = abs(r["diff"]) <= 0.05 + 1e-9  # paper reports one decimal
                per.append(r)
            per_df = pd.DataFrame(per)
            avg = {"model": model, "split": split, "dataset": "Average"}
            for c in per_df.columns:
                if c in ("n", "n_supported", "n_unsupported", "tn", "fp", "fn", "tp", "inference_seconds", "n_batch_fallbacks"):
                    avg[c] = per_df[c].sum()
                elif c in ("peak_gpu_mb_allocated", "peak_gpu_mb_reserved"):
                    avg[c] = per_df[c].max()
                elif c not in avg and pd.api.types.is_numeric_dtype(per_df[c]) and c not in ("matches_paper", "one_flip_max"):
                    avg[c] = per_df[c].mean()  # macro average, as in the paper
            avg["latency_ms_mean"] = 1000 * avg["inference_seconds"] / avg["n"]
            if split == "test":
                avg["paper_bacc"] = sum(paper["bacc"]) / len(paper["bacc"])
            pooled = {"model": model, "split": split, "dataset": "Pooled",
                      **scores(pd.concat(preds.values(), ignore_index=True))}
            rows += per + [avg, pooled]

            envs = [m["env"] for m in metas.values()]
            differing = {k: sorted({str(e.get(k)) for e in envs}) for k in envs[0]
                         if len({str(e.get(k)) for e in envs}) > 1}
            s = {"model": model, "split": split, "datasets_done": f"{len(preds)}/{len(DATASETS)}",
                 "n": int(avg["n"]), "bacc_macro": avg["bacc"], "macro_f1_mean": avg["macro_f1"],
                 "inference_seconds": avg["inference_seconds"], "ms_per_claim": avg["latency_ms_mean"],
                 "load_seconds": full_runs[0]["load_seconds"] if full_runs else None,
                 "weights_gpu_mb": full_runs[0]["weights_gpu_mb"] if full_runs else None,
                 "peak_gpu_mb_allocated": avg.get("peak_gpu_mb_allocated"),
                 "batch_fallbacks": int(per_df.n_batch_fallbacks.fillna(0).sum()),
                 "gpu": ", ".join(sorted(per_df.gpu.unique())),
                 "env_differences": json.dumps(differing) if differing else ""}
            if split == "test":
                s.update({"paper_row_mean": avg["paper_bacc"], "paper_printed_average": paper["printed_average"],
                          "diff_vs_row_mean": avg["bacc"] - avg["paper_bacc"],
                          "datasets_matching_paper": f"{int(per_df.matches_paper.sum())}/{len(per_df)}"})
                test_bacc[model] = per_df.set_index("dataset").bacc.to_dict() | {"Average": avg["bacc"]}
            if differing:
                print(f"WARNING {model}/{split}: environment differs between datasets: {differing}")
            summary.append(s)

    if not rows:
        raise SystemExit(f"no finished results under {args.results}")
    pd.DataFrame(rows).round(3).to_csv(out / "per_dataset.csv", index=False)
    summary_df = pd.DataFrame(summary).round(3)
    summary_df.to_csv(out / "summary.csv", index=False)

    # Test-split comparison with the paper (markdown, for the report)
    names = list(test_bacc)
    lines = ["| Dataset | " + " | ".join(f"{m} (ours) | paper | diff" for m in names) + " |",
             "|---|" + "---:|" * (3 * len(names))]
    for name in DATASETS + ["Average"]:
        cells = []
        for m in names:
            ours = test_bacc[m].get(name)
            pap = (sum(CFG["paper_table2"][m]["bacc"]) / 10 if name == "Average"
                   else dict(zip(DATASETS, CFG["paper_table2"][m]["bacc"]))[name])
            cells += ["–", f"{pap:.1f}", "–"] if ours is None else [f"{ours:.2f}", f"{pap:.2f}" if name == "Average" else f"{pap:.1f}", f"{ours - pap:+.2f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Paper 'Average' = mean of the Table 2 row. Printed Avg column: " +
                 ", ".join(f"{m} {CFG['paper_table2'][m]['printed_average']}" for m in names) + ".")
    (out / "comparison_test.md").write_text("\n".join(lines), encoding="utf-8")

    if args.legacy and (args.results / "roberta-large" / "test").exists():
        preds, _ = load_run(args.results / "roberta-large" / "test")
        if len(preds) == len(DATASETS):
            ours = pd.concat(preds.values()).sort_values("idx").reset_index(drop=True)  # parquet order
            old = pd.read_csv(args.legacy)
            aligned = len(old) == len(ours) and (old.dataset.values == ours.dataset.values).all() \
                and (old.label.values == ours.label.values).all()
            check = {"aligned": bool(aligned)}
            if aligned:
                flips = old.pred.values != ours.pred.values
                check.update({"n": len(ours), "prediction_flips": int(flips.sum()),
                              "flips_per_dataset": ours[flips].dataset.value_counts().to_dict(),
                              "max_abs_prob_diff": float((old.prob - ours.prob).abs().max()),
                              "mean_abs_prob_diff": float((old.prob - ours.prob).abs().mean())})
            (out / "hardware_check.json").write_text(json.dumps(check, indent=2))
            print("hardware check (laptop vs this run):", json.dumps(check))

    pd.set_option("display.width", 200)
    print(summary_df.drop(columns=["env_differences"]).to_string(index=False))
    print()
    print((out / "comparison_test.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
