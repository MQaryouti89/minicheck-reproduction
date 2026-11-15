"""Phase 5: error analysis of the three checkers and the frozen cascade on the test split (CPU only).

    python scripts/error_analysis.py              # everything that needs only the saved predictions
    python scripts/error_analysis.py --with-text  # + claim types and failure examples (needs access to the
                                                  #   gated LLM-AggreFact files: `hf auth login` or HF_TOKEN)

Inputs (read-only): Phase 3 test predictions (results/kaggle-t4/minicheck_results), the frozen delta
(results/phase4/selected_thresholds.json); with --with-text, the pinned test parquet (configs/experiment.json).
Outputs: results/phase5/*.csv, *.png, error_analysis.md. Claim and document text is never written there;
failure examples with text go to private/phase5_examples.md, which is excluded from git (gated dataset).

Claim-type definitions (fixed before any claim text was examined; simple, transparent heuristics):
  numeric   a digit, or a number word (NUMBER_WORDS)
  negation  a negation word (NEGATION_WORDS) or the contraction "n't"
  complex   more than 25 words, or at least two clause markers (CLAUSE_MARKERS)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import phase4  # noqa: E402  (same data loading, cascade rule and BAcc code as Phase 4)
from phase4 import COL, DATASETS, NAMES, macro_bacc  # noqa: E402

OUT = ROOT / "results" / "phase5"
PRIVATE = ROOT / "private"
FROZEN = ROOT / "results" / "phase4" / "selected_thresholds.json"
SEED, REPS = 504, 1000

NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "hundreds", "thousand", "thousands",
                "million", "millions", "billion", "billions", "percent", "dozen", "half", "twice", "double"}
NEGATION_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot"}
CLAUSE_MARKERS = {"and", "but", "or", "while", "whereas", "which", "who", "because", "although"}
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def claim_features(claim):
    text = str(claim).lower()
    words = WORD.findall(text)
    return {
        "n_words": len(text.split()),
        "numeric": bool(re.search(r"\d", text)) or any(w in NUMBER_WORDS for w in words),
        "negation": "n't" in text or any(w in NEGATION_WORDS for w in words),
        "complex": len(text.split()) > 25 or sum(w in CLAUSE_MARKERS for w in words) >= 2,
    }


# ---------------------------------------------------------------- helpers
def pooled_bacc(y, pred):
    c = pred == y
    return 50 * (c[y == 1].mean() + c[y == 0].mean()) if (y == 1).any() and (y == 0).any() else np.nan


def boot_pooled(D, mask, preds, reps=REPS, seed=SEED):
    """BAcc within a claim group, CI by resampling claims within each dataset (keeps the group's dataset mix).
    preds: {name: prediction array}. Returns {name: (point, lo, hi)} and the paired Flan - RoBERTa difference."""
    rng = np.random.default_rng(seed)
    idx_by_ds = [np.flatnonzero(mask & (D.dataset.values == d)) for d in DATASETS]
    idx_by_ds = [i for i in idx_by_ds if len(i)]
    y = D.label.values
    reps_v = {k: [] for k in preds}
    for _ in range(reps):
        s = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in idx_by_ds])
        for k, p in preds.items():
            reps_v[k].append(pooled_bacc(y[s], p[s]))
    m = np.flatnonzero(mask)
    out = {k: (pooled_bacc(y[m], p[m]), *np.nanpercentile(reps_v[k], [2.5, 97.5])) for k, p in preds.items()}
    diff = np.array(reps_v["Flan-T5"]) - np.array(reps_v["RoBERTa"])
    out["Flan-T5 - RoBERTa"] = (out["Flan-T5"][0] - out["RoBERTa"][0], *np.nanpercentile(diff, [2.5, 97.5]))
    return out


def figure_setup(w=7.5, h=4.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(COL["surface"]); ax.set_facecolor(COL["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["grid"])
    ax.tick_params(colors=COL["muted"], labelsize=8)
    ax.set_axisbelow(True)
    return fig, ax, plt


def md(df, fmt=2):
    return phase4.md_table(df, fmt)


# ---------------------------------------------------------------- analyses that need only predictions
def prediction_analyses(D, delta):
    y = D.label.values
    cR, cD, cF = (D.pred_R.values == y), (D.pred_D.values == y), (D.pred_F.values == y)
    pred_c, cost_c, d = phase4.cascade(D, delta)
    cC = pred_c == y
    res = {}

    # 1. agreement between RoBERTa and Flan-T5, per dataset
    rows = []
    for name in DATASETS + ["All"]:
        m = np.ones(len(D), bool) if name == "All" else D.dataset.values == name
        n = m.sum()
        rows.append({"dataset": name, "claims": int(n),
                     "both right %": 100 * (cR & cF)[m].mean(),
                     "only RoBERTa wrong %": 100 * (~cR & cF)[m].mean(),
                     "only Flan-T5 wrong %": 100 * (cR & ~cF)[m].mean(),
                     "both wrong %": 100 * (~cR & ~cF)[m].mean(),
                     "all three wrong %": 100 * (~cR & ~cD & ~cF)[m].mean()})
    res["agreement_by_dataset"] = pd.DataFrame(rows)

    # 2. false positives / false negatives (positive = supported): rates per model and dataset
    rows = []
    for key, name, pred in [("R", "RoBERTa", D.pred_R.values), ("D", "DeBERTa", D.pred_D.values),
                            ("F", "Flan-T5", D.pred_F.values), ("C", "Cascade", pred_c)]:
        for ds in DATASETS + ["All"]:
            m = np.ones(len(D), bool) if ds == "All" else D.dataset.values == ds
            yy, pp = y[m], pred[m]
            rows.append({"model": name, "dataset": ds,
                         "FP (missed error)": int(((pp == 1) & (yy == 0)).sum()),
                         "FN (false alarm)": int(((pp == 0) & (yy == 1)).sum()),
                         "FP rate % (of unsupported)": 100 * ((pp == 1) & (yy == 0)).sum() / max((yy == 0).sum(), 1),
                         "FN rate % (of supported)": 100 * ((pp == 0) & (yy == 1)).sum() / max((yy == 1).sum(), 1)})
    res["fp_fn_by_dataset"] = pd.DataFrame(rows)

    # 3. confident RoBERTa errors that the gate keeps (not deferred)
    conf = np.abs(D.p_R.values - 0.5)
    bins = [0, delta, 0.3, 0.4, 0.45, 0.5001]
    labels = [f"[0, {delta}) deferred", f"[{delta}, 0.30)", "[0.30, 0.40)", "[0.40, 0.45)", "[0.45, 0.50]"]
    b = pd.cut(conf, bins, labels=labels, right=False)
    t = pd.DataFrame({"bin": b, "R_wrong": ~cR, "F_fixes": ~cR & cF, "FP": (D.pred_R.values == 1) & (y == 0),
                      "FN": (D.pred_R.values == 0) & (y == 1)})
    g = t.groupby("bin", observed=False).agg(claims=("R_wrong", "size"), roberta_errors=("R_wrong", "sum"),
                                             flan_would_fix=("F_fixes", "sum"), of_which_FP=("FP", "sum"),
                                             of_which_FN=("FN", "sum"))
    g["share of all RoBERTa errors %"] = 100 * g.roberta_errors / g.roberta_errors.sum()
    g["RoBERTa error rate %"] = 100 * g.roberta_errors / g.claims
    res["confident_errors"] = g.reset_index(names="|p_RoBERTa - 0.5|")

    # 4. document length: errors and cost
    lb = pd.cut(D.chunks_R, [0, 1, 3, 7, 10**6], labels=["1 chunk", "2-3", "4-7", "8+"])
    t = pd.DataFrame({"len": lb, "eR": ~cR, "eF": ~cF, "eC": ~cC, "d": d, "lR": D.lat_R.values,
                      "lF": D.lat_F.values, "lC": cost_c})
    g = t.groupby("len", observed=False).agg(claims=("eR", "size"), roberta_error=("eR", "mean"),
                                             flan_error=("eF", "mean"), cascade_error=("eC", "mean"),
                                             deferred=("d", "mean"), roberta_ms=("lR", "mean"),
                                             flan_ms=("lF", "mean"), cascade_s=("lC", "sum"), flan_s=("lF", "sum"))
    for c in ("roberta_error", "flan_error", "cascade_error", "deferred"):
        g[c] *= 100
    g["share of claims %"] = 100 * g.claims / g.claims.sum()
    g["share of cascade time %"] = 100 * g.cascade_s / g.cascade_s.sum()
    g["share of Flan-T5 time %"] = 100 * g.flan_s / g.flan_s.sum()
    g[["cascade_s", "flan_s"]] /= 1000
    res["length_and_cost"] = g.reset_index(names="document length (RoBERTa chunks)")

    # 5. per dataset: models, cascade, deferral
    rows = []
    for ds in DATASETS:
        m = D.dataset.values == ds
        rows.append({"dataset": ds, "RoBERTa": pooled_bacc(y[m], D.pred_R.values[m]),
                     "DeBERTa": pooled_bacc(y[m], D.pred_D.values[m]), "Flan-T5": pooled_bacc(y[m], D.pred_F.values[m]),
                     "Cascade": pooled_bacc(y[m], pred_c[m]), "deferred %": 100 * d[m].mean(),
                     "best single model": max(("RoBERTa", "DeBERTa", "Flan-T5"),
                                              key=lambda k: pooled_bacc(y[m], D[f"pred_{k[0]}"].values[m]))})
    res["per_dataset"] = pd.DataFrame(rows)
    return res, (pred_c, cost_c, d)


def prediction_figures(D, delta, res):
    # confidence bins: RoBERTa vs Flan-T5 accuracy (test), with the frozen delta
    conf = np.abs(D.p_R.values - 0.5)
    edges = np.round(np.arange(0, 0.55, 0.05), 2)
    b = np.clip(np.digitize(conf, edges) - 1, 0, len(edges) - 2)
    y = D.label.values
    accR = [100 * (D.pred_R.values == y)[b == i].mean() for i in range(len(edges) - 1)]
    accF = [100 * (D.pred_F.values == y)[b == i].mean() for i in range(len(edges) - 1)]
    share = [100 * (b == i).mean() for i in range(len(edges) - 1)]
    mids = edges[:-1] + 0.025
    fig, ax, plt = figure_setup()
    ax.grid(True, axis="y", color=COL["grid"], linewidth=0.6)
    ax.axvspan(0, delta, color=COL["cascade"], alpha=0.07, lw=0)
    ax.text(delta / 2, 47, f"deferred\n(δ = {delta})", ha="center", fontsize=7, color=COL["muted"])
    ax.plot(mids, accR, "o-", color=COL["cascade"], lw=2, ms=5, label="RoBERTa")
    ax.plot(mids, accF, "s-", color=COL["selected"], lw=2, ms=5, label="Flan-T5")
    for x, s in zip(mids, share):
        ax.annotate(f"{s:.0f}%", (x, 44), ha="center", fontsize=6.5, color=COL["muted"])
    ax.set_ylim(42, 88)
    ax.set_xlabel("RoBERTa confidence |p − 0.5| (bins of 0.05); % = share of test claims in the bin",
                  color=COL["muted"], fontsize=8.5)
    ax.set_ylabel("accuracy on the claims in the bin (%)", color=COL["muted"], fontsize=8.5)
    ax.set_title("Test: where Flan-T5 is better than RoBERTa", color=COL["text"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=COL["text"])
    fig.tight_layout(); fig.savefig(OUT / "fig_confidence_bins.png", facecolor=COL["surface"]); plt.close(fig)

    # error categories by dataset
    a = res["agreement_by_dataset"]
    a = a[a.dataset != "All"].iloc[::-1]
    fig, ax, plt = figure_setup(7.5, 4.0)
    ax.grid(True, axis="x", color=COL["grid"], linewidth=0.6)
    left = np.zeros(len(a))
    for col, colour, lab in [("only RoBERTa wrong %", COL["cascade"], "only RoBERTa wrong (Flan-T5 right)"),
                             ("only Flan-T5 wrong %", COL["selected"], "only Flan-T5 wrong (RoBERTa right)"),
                             ("both wrong %", COL["base"], "both wrong")]:
        ax.barh(range(len(a)), a[col], left=left, color=colour, height=0.6, label=lab, edgecolor=COL["surface"], lw=1)
        left += a[col].values
    for i, v in enumerate(left):
        ax.annotate(f"{v:.0f}%", (v, i), xytext=(4, -3), textcoords="offset points", fontsize=7, color=COL["muted"])
    ax.set_yticks(range(len(a))); ax.set_yticklabels(a.dataset, fontsize=8, color=COL["text"])
    ax.tick_params(left=False)
    ax.set_xlabel("% of the dataset's test claims", color=COL["muted"], fontsize=8.5)
    ax.set_title("Test: who is wrong, per dataset", color=COL["text"], fontsize=10, loc="left")
    ax.set_xlim(0, 72)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=COL["text"])
    fig.tight_layout(); fig.savefig(OUT / "fig_error_categories.png", facecolor=COL["surface"]); plt.close(fig)

    # document length: share of claims vs share of time
    g = res["length_and_cost"]
    fig, ax, plt = figure_setup(6.5, 3.6)
    ax.grid(True, axis="y", color=COL["grid"], linewidth=0.6)
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w / 2, g["share of claims %"], w, color=COL["base"], label="share of test claims")
    ax.bar(x + w / 2, g["share of cascade time %"], w, color=COL["cascade"], label="share of cascade time (simulated)")
    for xi, (s1, s2) in enumerate(zip(g["share of claims %"], g["share of cascade time %"])):
        ax.annotate(f"{s1:.0f}%", (xi - w / 2, s1), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=COL["muted"])
        ax.annotate(f"{s2:.0f}%", (xi + w / 2, s2), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=COL["muted"])
    ax.set_xticks(x); ax.set_xticklabels(g.iloc[:, 0], fontsize=8, color=COL["text"])
    ax.set_xlabel("document length (number of RoBERTa chunks)", color=COL["muted"], fontsize=8.5)
    ax.set_ylabel("%", color=COL["muted"], fontsize=8.5)
    ax.set_title("Test: long documents are few but expensive", color=COL["text"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=COL["text"])
    fig.tight_layout(); fig.savefig(OUT / "fig_length_cost.png", facecolor=COL["surface"]); plt.close(fig)


# ---------------------------------------------------------------- claim types and examples (needs the gated text)
def load_text(data_dir=None):
    import run_checker as rc
    df = rc.load_split("test", data_dir, None)
    return df.set_index("idx")[["dataset", "label", "doc", "claim"]]


def text_analyses(D, T, pred_c, d, delta):
    T = T.loc[D.index]
    assert (T.label.values == D.label.values).all() and (T.dataset.values == D.dataset.values).all(), \
        "text file does not match the prediction ids"
    F = pd.DataFrame([claim_features(c) for c in T.claim], index=D.index)
    F["doc_chars"] = T.doc.str.len().values
    F["dataset"] = D.dataset.values
    F.reset_index(names="idx").to_csv(OUT / "claim_features.csv", index=False)  # flags only, no text

    preds = {"RoBERTa": D.pred_R.values, "Flan-T5": D.pred_F.values, "Cascade": pred_c}
    rows = []
    for feat in ("numeric", "negation", "complex"):
        for present in (True, False):
            m = (F[feat] == present).values
            r = boot_pooled(D, m, preds)
            rows.append({"claim type": feat, "present": present, "claims": int(m.sum()),
                         "share %": 100 * m.mean(),
                         **{f"{k} BAcc": v[0] for k, v in r.items()},
                         **{f"{k} CI": f"[{v[1]:.1f}, {v[2]:.1f}]" for k, v in r.items()},
                         "deferred %": 100 * d[m].mean()})
    types = pd.DataFrame(rows)
    prevalence = F.groupby("dataset")[["numeric", "negation", "complex"]].mean().mul(100).loc[DATASETS]
    prevalence["claim words (median)"] = F.groupby("dataset").n_words.median().loc[DATASETS]
    return F, types, prevalence


def within_dataset(D, F, pred_c, min_per_class=20):
    """Claim-type effects inside each dataset, to remove the dataset-mix confound of the pooled table.
    A dataset is used for a flag if both groups (flag present / absent) have >= min_per_class supported and
    unsupported claims. Reports BAcc(present) - BAcc(absent) per model, and the Flan-T5 - RoBERTa gap per group."""
    y = D.label.values
    preds = {"RoBERTa": D.pred_R.values, "Flan-T5": D.pred_F.values, "Cascade": pred_c}
    rows = []
    for feat in ("numeric", "negation", "complex"):
        for ds in DATASETS:
            m = D.dataset.values == ds
            g1, g0 = m & F[feat].values, m & ~F[feat].values
            ok = all(((y[g] == c).sum() >= min_per_class) for g in (g1, g0) for c in (0, 1))
            if not ok:
                continue
            r = {"claim type": feat, "dataset": ds, "claims with flag": int(g1.sum()), "claims without": int(g0.sum())}
            for k, p in preds.items():
                r[f"{k}: with - without"] = pooled_bacc(y[g1], p[g1]) - pooled_bacc(y[g0], p[g0])
            r["Flan-T5 - RoBERTa, with flag"] = pooled_bacc(y[g1], preds["Flan-T5"][g1]) - pooled_bacc(y[g1], preds["RoBERTa"][g1])
            r["Flan-T5 - RoBERTa, without"] = pooled_bacc(y[g0], preds["Flan-T5"][g0]) - pooled_bacc(y[g0], preds["RoBERTa"][g0])
            rows.append(r)
    per = pd.DataFrame(rows)
    if per.empty:
        return per, pd.DataFrame()
    cols = [c for c in per.columns if " - " in c]
    summ = per.groupby("claim type").agg(datasets=("dataset", "size"),
                                         **{f"mean {c}": (c, "mean") for c in cols},
                                         **{f"datasets with {c} < 0": (c, lambda s: int((s < 0).sum())) for c in cols[:3]})
    return per, summ.reindex(["numeric", "negation", "complex"]).reset_index()


def short_documents(D, T, limit=200):
    """Claims labelled supported whose document is very short: does a missing-evidence pattern explain errors?"""
    y = D.label.values
    L = T.loc[D.index].doc.str.len().values
    all_wrong = (D.pred_R.values != y) & (D.pred_D.values != y) & (D.pred_F.values != y)
    rows = []
    for ds in DATASETS + ["All"]:
        m = np.ones(len(D), bool) if ds == "All" else D.dataset.values == ds
        s = m & (L < limit)
        sy = s & (y == 1)
        rows.append({"dataset": ds, f"docs < {limit} chars": int(s.sum()), "of which supported": int(sy.sum()),
                     "all three wrong on those %": 100 * all_wrong[sy].mean() if sy.any() else np.nan,
                     "all three wrong on the rest %": 100 * all_wrong[m & ~sy].mean()})
    return pd.DataFrame(rows)


def select_examples(D, T, F, pred_c, d):
    """Deterministic: per category, the claims with the shortest documents (readable), one per dataset."""
    y = D.label.values
    cR, cF, cC = D.pred_R.values == y, D.pred_F.values == y, pred_c == y
    conf = np.abs(D.p_R.values - 0.5)
    cats = {
        "A. Confident RoBERTa error, kept (not deferred); Flan-T5 would be right": ~d & ~cR & cF & (conf >= 0.45),
        "B. Deferred and broken: RoBERTa right, Flan-T5 wrong": d & cR & ~cF,
        "C. All models wrong with high confidence": ~cR & ~cF & ~(D.pred_D.values == y) & (conf >= 0.45),
        "D. Numeric claim, cascade wrong": F.numeric.values & ~cC,
        "E. Negated claim, cascade wrong": F.negation.values & ~cC,
        "F. Complex claim, cascade wrong": F.complex.values & ~cC,
    }
    chosen = []
    doc_len = T.loc[D.index].doc.str.len().values
    for cat, m in cats.items():
        cand = pd.DataFrame({"idx": D.index[m], "dataset": D.dataset.values[m], "len": doc_len[m]})
        cand = cand[cand.len <= 1500].sort_values(["len", "idx"]).drop_duplicates("dataset").head(2)
        for i in cand.idx:
            chosen.append((cat, i))
    rows = []
    for cat, i in chosen:
        r = D.loc[i]
        rows.append({"category": cat, "idx": i, "dataset": r.dataset, "label": int(r.label),
                     "p_RoBERTa": round(r.p_R, 3), "p_DeBERTa": round(r.p_D, 3), "p_Flan-T5": round(r.p_F, 3),
                     "deferred": bool(abs(r.p_R - 0.5) < json.loads(FROZEN.read_text())["primary"]["delta"]),
                     "numeric": bool(F.loc[i, "numeric"]), "negation": bool(F.loc[i, "negation"]),
                     "complex": bool(F.loc[i, "complex"])})
    ex = pd.DataFrame(rows)
    ex.to_csv(OUT / "examples_idx.csv", index=False)  # no text
    PRIVATE.mkdir(exist_ok=True)
    lines = ["# Phase 5 failure examples (PRIVATE: contains gated LLM-AggreFact text; do not commit or share)", "",
             "Label 1 = supported, 0 = unsupported. p = probability of 'supported'. Selected deterministically "
             "(shortest documents, one per dataset per category) by scripts/error_analysis.py.", ""]
    for _, r in ex.iterrows():
        t = T.loc[r.idx]
        lines += [f"## {r.category}", "",
                  f"- idx `{r.idx}`, {r.dataset}, gold label **{r.label}**; p_RoBERTa {r.p_RoBERTa}, "
                  f"p_DeBERTa {r['p_DeBERTa']}, p_Flan-T5 {r['p_Flan-T5']}; deferred: {r.deferred}",
                  f"- claim: {t.claim}", "", "document:", "", "> " + str(t.doc).replace("\n", "\n> "), ""]
    (PRIVATE / "phase5_examples.md").write_text("\n".join(lines), encoding="utf-8")
    return ex


# ---------------------------------------------------------------- report
def write_md(res, delta, types=None, prevalence=None, examples=None, extra=None):
    lines = ["# Phase 5 error analysis (test split)", "",
             f"Models: Phase 3 predictions on the Kaggle T4. Cascade: frozen δ = {delta} (Phase 4). "
             "Positive class = supported, so FP = unsupported claim accepted (missed error), "
             "FN = supported claim rejected (false alarm).", ""]
    titles = {"agreement_by_dataset": "1. Agreement between RoBERTa and Flan-T5",
              "fp_fn_by_dataset": "2. False positives and false negatives (All = pooled)",
              "confident_errors": "3. RoBERTa errors by confidence: what the gate can and cannot reach",
              "length_and_cost": "4. Document length: errors (%) and cost",
              "per_dataset": "5. Per-dataset BAcc (pooled within the dataset) and deferral"}
    for k, title in titles.items():
        v = res[k]
        if k == "fp_fn_by_dataset":
            v = v[v.dataset == "All"].drop(columns="dataset")
        lines += [f"## {title}", "", md(v), ""]
    if types is not None:
        lines += ["## 6. Claim types (heuristic flags; definitions in scripts/error_analysis.py)", "",
                  "BAcc pooled over the claims of the group; 95% CI from 1,000 resamples within datasets. "
                  "Groups differ in dataset mix, so differences between groups are associations, not causes.", "",
                  md(types), "", "Share of claims with each flag, per dataset (%):", "",
                  md(prevalence.reset_index(names="dataset"), 1), ""]
    if extra is not None:
        lines += ["### 6b. Within datasets (removes the dataset-mix confound)", "",
                  "Datasets where both groups have >= 20 supported and >= 20 unsupported claims. "
                  "Differences in BAcc points; unweighted means over the eligible datasets.", "",
                  md(extra["within_summary"]), "", md(extra["within"]), "",
                  "## 6c. Very short documents labelled supported (missing evidence?)", "",
                  "Seen in the failure examples; checked here on all test claims (document length only, no text).", "",
                  md(extra["short"]), ""]
    if examples is not None:
        lines += ["## 7. Failure examples (ids only; text in private/phase5_examples.md, not committed)", "",
                  "Selected deterministically: in each category, the shortest documents (readable), one per dataset. "
                  "This favours very short documents, so the examples illustrate error types; they are not a "
                  "random sample and do not show how often each type occurs.", "",
                  md(examples, 3), ""]
    (OUT / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-text", action="store_true", help="claim types and examples (needs the gated test file)")
    ap.add_argument("--data-dir", type=Path, help="read test.parquet from here instead of Hugging Face")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    delta = json.loads(FROZEN.read_text())["primary"]["delta"]
    D = phase4.load("test")
    phase4.check_baselines(D, "test")
    res, (pred_c, cost_c, d) = prediction_analyses(D, delta)
    assert abs(macro_bacc(D, pred_c)[0] - 74.50735545727892) < 1e-9, "cascade BAcc differs from Phase 4"
    for k, v in res.items():
        v.round(3).to_csv(OUT / f"{k}.csv", index=False)
    prediction_figures(D, delta, res)
    types = prevalence = examples = None
    if args.with_text:
        T = load_text(args.data_dir)
        F, types, prevalence = text_analyses(D, T, pred_c, d, delta)
        types.round(3).to_csv(OUT / "claim_types.csv", index=False)
        prevalence.round(2).to_csv(OUT / "claim_type_prevalence.csv")
        within, within_summary = within_dataset(D, F, pred_c)
        within.round(3).to_csv(OUT / "claim_types_within_dataset.csv", index=False)
        within_summary.round(3).to_csv(OUT / "claim_types_within_dataset_summary.csv", index=False)
        short = short_documents(D, T)
        short.round(2).to_csv(OUT / "short_documents.csv", index=False)
        examples = select_examples(D, T, F, pred_c, d)
        extra = {"within": within, "within_summary": within_summary, "short": short}
    else:
        extra = None
    write_md(res, delta, types, prevalence, examples, extra)
    sys.stdout.reconfigure(encoding="utf-8")
    print((OUT / "error_analysis.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
