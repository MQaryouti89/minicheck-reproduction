"""Phase 4: adaptive RoBERTa -> Flan-T5 cascade, simulated from the saved Phase 3 predictions (CPU only).

The plan (defer rule, delta grid, selection rule, seeds, success criteria) is fixed in configs/phase4.json.

    python scripts/phase4.py select     # dev only: delta curve, choose delta, freeze it
    python scripts/phase4.py evaluate   # test, once: reads the frozen delta, scores every approach
    python scripts/phase4.py measured   # after the Kaggle cascade run: measured vs simulated

Inputs (read-only): results/kaggle-t4/minicheck_results/<model>/<split>/<dataset>.csv
Outputs:            results/phase4/

"Simulated" = predictions and per-claim latencies of each model measured separately in Phase 3 and
combined here. It is not an end-to-end measurement; `measured` compares it with one.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import scores  # noqa: E402  (same metric code as the Phase 3 baselines)

CFG = json.loads((ROOT / "configs" / "experiment.json").read_text())
P4_FILE = ROOT / "configs" / "phase4.json"
P4 = json.loads(P4_FILE.read_text())
DATASETS = CFG["datasets"]
RESULTS = ROOT / "results" / "kaggle-t4" / "minicheck_results"
OUT = ROOT / "results" / "phase4"
FROZEN = OUT / "selected_thresholds.json"
MODELS = {"R": "roberta-large", "D": "deberta-v3-large", "F": "flan-t5-large"}
NAMES = {"R": "RoBERTa", "D": "DeBERTa", "F": "Flan-T5"}
THRESHOLD = CFG["threshold"]


def sha(path):
    """Return the first 16 hex characters of a file's SHA-256 (used to detect changes to frozen inputs)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- data
def load(split, results=None):
    """One row per claim, columns per model (R/D/F). Stops if ids, labels or datasets differ between models."""
    results = results or RESULTS
    frames = {}
    for key, model in MODELS.items():
        df = pd.concat([pd.read_csv(results / model / split / f"{d}.csv") for d in DATASETS])
        frames[key] = df.set_index("idx").sort_index()
    base = frames["R"]
    for key, df in frames.items():
        same = (base.index.equals(df.index) and (base.label.values == df.label.values).all()
                and (base.dataset.values == df.dataset.values).all())
        if not same:
            sys.exit(f"{split}: idx/label/dataset differ between {MODELS['R']} and {MODELS[key]}; not combining")
        if not (df.pred.values == (df.prob.values > THRESHOLD)).all():
            sys.exit(f"{split}/{MODELS[key]}: pred is not prob > {THRESHOLD}")
    if base.dataset.value_counts().to_dict() != CFG["expected_counts"][split]:
        sys.exit(f"{split}: claim counts differ from configs/experiment.json")
    out = pd.DataFrame({"dataset": base.dataset, "label": base.label.astype(int)}, index=base.index)
    for key, df in frames.items():
        out[f"p_{key}"] = df.prob
        out[f"pred_{key}"] = df.pred.astype(int)
        out[f"lat_{key}"] = df.latency_ms
        out[f"chunks_{key}"] = df.n_chunks
    return out


def deltas():
    """Return the delta grid fixed in configs/phase4.json (start, stop, step), rounded to avoid float drift."""
    g = P4["cascade"]["delta_grid"]
    n = int(round((g["stop"] - g["start"]) / g["step"])) + 1
    return np.round(g["start"] + g["step"] * np.arange(n), 6)


def deferred(D, delta):
    """Boolean mask of the claims the cascade sends to Flan-T5: |p_RoBERTa - 0.5| < delta."""
    return (D.p_R - 0.5).abs().values < delta


# ---------------------------------------------------------------- approaches: (prediction, cost in ms, deferred mask)
def cascade(D, delta):
    """Simulate the RoBERTa -> Flan-T5 cascade at one delta.

    Returns (predictions, per-claim cost in ms, deferral mask). The cost is RoBERTa's measured
    latency plus Flan-T5's measured latency for deferred claims.
    """
    d = deferred(D, delta)
    return np.where(d, D.pred_F, D.pred_R), D.lat_R.values + d * D.lat_F.values, d


def single(D, key):
    """Predictions and per-claim latency of one checker alone (key 'R', 'D' or 'F'); no deferral mask."""
    return D[f"pred_{key}"].values, D[f"lat_{key}"].values, None


def ensemble(D):
    """Majority vote of the three checkers; its per-claim cost is the sum of the three latencies."""
    votes = D.pred_R.values + D.pred_D.values + D.pred_F.values
    return (votes >= 2).astype(int), (D.lat_R + D.lat_D + D.lat_F).values, None


def oracle(D):
    """Upper bound: defer exactly the claims that RoBERTa gets wrong and Flan-T5 gets right."""
    d = ((D.pred_R != D.label) & (D.pred_F == D.label)).values
    return np.where(d, D.pred_F, D.pred_R), D.lat_R.values + d * D.lat_F.values, d


def random_deferral(D, n_per_dataset, draws, seed):
    """`draws` random deferral masks with exactly n_per_dataset[dataset] deferred claims in each dataset."""
    rng = np.random.default_rng(seed)
    pos = {name: np.flatnonzero(D.dataset.values == name) for name in DATASETS}
    masks = np.zeros((draws, len(D)), dtype=bool)
    for i in range(draws):
        for name, idx in pos.items():
            masks[i, rng.choice(idx, size=n_per_dataset[name], replace=False)] = True
    return masks


def macro_bacc(D, pred):
    """Paper's aggregation: BAcc per dataset (in %), unweighted mean. Returns (mean, {dataset: bacc})."""
    y, c = D.label.values, (pred == D.label.values)
    per = {}
    for name in DATASETS:
        m = D.dataset.values == name
        per[name] = 50 * (c[m & (y == 1)].mean() + c[m & (y == 0)].mean())
    return float(np.mean(list(per.values()))), per


def describe(D, name, pred, cost, d, flan_total):
    """All agreed metrics for one approach. P/R/F1 are macro averages over datasets (as Phase 3's 'Average')."""
    df = pd.DataFrame({"label": D.label.values, "pred": pred, "dataset": D.dataset.values})
    per = {ds: scores(g) for ds, g in df.groupby("dataset", sort=False)}
    row = {"approach": name}
    for k in ["bacc", "precision_supported", "recall_supported", "f1_supported",
              "precision_unsupported", "recall_unsupported", "f1_unsupported", "macro_f1"]:
        row[k] = float(np.mean([per[ds][k] for ds in DATASETS]))
    for k in ["tn", "fp", "fn", "tp"]:
        row[k] = int(sum(per[ds][k] for ds in DATASETS))
    row["deferred_pct"] = 100 * d.mean() if d is not None else np.nan
    row["ms_per_claim"] = cost.mean()
    row["total_seconds"] = cost.sum() / 1000
    row["speedup_vs_flan"] = flan_total / (cost.sum() / 1000)
    row["time_vs_flan"] = (cost.sum() / 1000) / flan_total
    return row, {ds: per[ds]["bacc"] for ds in DATASETS}


# ---------------------------------------------------------------- paired bootstrap
def bootstrap(D, correct, costs, replicates, seed, chunk=500):
    """Paired bootstrap of macro BAcc: claims resampled with replacement within each dataset, the same resamples
    for every approach. `correct` maps approach -> per-claim correctness (0/1, or an expected value in [0, 1]);
    BAcc is linear in it, so an expected correctness gives the expected BAcc over random draws exactly.
    Returns {approach: macro BAcc per replicate}, {approach: total cost (s) per replicate}."""
    rng = np.random.default_rng(seed)
    y = D.label.values
    per_ds = {a: np.zeros((replicates, len(DATASETS))) for a in correct}
    tot = {a: np.zeros(replicates) for a in costs}
    for j, name in enumerate(DATASETS):
        m = D.dataset.values == name
        yk, n = y[m], int(m.sum())
        for s in range(0, replicates, chunk):
            b = min(chunk, replicates - s)
            W = rng.multinomial(n, np.full(n, 1 / n), size=b).astype(float)  # resample counts, b x n
            P = W @ (yk == 1)
            N = n - P
            for a, c in correct.items():
                ck = c[m]
                per_ds[a][s:s + b, j] = 50 * (W @ (ck * (yk == 1)) / P + W @ (ck * (yk == 0)) / N)
            for a, cost in costs.items():
                tot[a][s:s + b] += W @ cost[m] / 1000
    return {a: v.mean(axis=1) for a, v in per_ds.items()}, tot


def ci(x):
    """Percentile 95% confidence interval (2.5th and 97.5th percentiles) of bootstrap replicates."""
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- figures
COL = {"text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb",
       "cascade": "#2a78d6", "selected": "#eb6834", "random": "#1baf7a", "base": "#52514e"}


def tradeoff_figure(path, title, curve, points, flan_ms, flan_bacc, extra=None, criteria=False):
    """BAcc (y) vs simulated ms per claim (x). curve: DataFrame(delta, ms_per_claim, bacc).
    points: list of (label, ms, bacc, colour, marker). extra: optional (DataFrame, label) second curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    fig.patch.set_facecolor(COL["surface"]); ax.set_facecolor(COL["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["grid"])
    ax.grid(True, color=COL["grid"], linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=COL["muted"], labelsize=8)
    ax.axhline(flan_bacc, color=COL["muted"], lw=0.8, ls="--")
    if extra is not None:
        ex, lab = extra
        ax.plot(ex.ms_per_claim, ex.bacc, color=COL["cascade"], lw=1.2, alpha=0.35, ls="--", label=lab)
    if curve is not None:
        ax.plot(curve.ms_per_claim, curve.bacc, color=COL["cascade"], lw=2, label="cascade, delta 0 to 0.5")
    for dv in (0.1, 0.3):
        r = (curve if curve is not None else extra[0])
        r = r.loc[np.isclose(r.delta, dv)]
        if len(r):
            ax.plot(r.ms_per_claim, r.bacc, "o", ms=3, color=COL["cascade"])
            ax.annotate(f"delta {dv}", (r.ms_per_claim.iloc[0], r.bacc.iloc[0]), textcoords="offset points",
                        xytext=(4, -10), fontsize=7, color=COL["muted"])
    for lab, x, yv, colour, marker, *off in points:
        ax.plot(x, yv, marker, ms=8, color=colour, mec=COL["surface"], mew=2, zorder=5)
        dx, dy, ha = off[0] if off else (7, 4, "left")
        ax.annotate(lab, (x, yv), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8, color=COL["text"])
    if criteria:  # success region for outcome 2 (point estimates; the test itself uses the CI)
        ax.axhline(flan_bacc - 0.5, color=COL["muted"], lw=0.8, ls=":")
        ax.axvline(0.8 * flan_ms, color=COL["muted"], lw=0.8, ls=":")
        ax.annotate("1.25x faster than Flan-T5", (0.8 * flan_ms, flan_bacc - 0.5), textcoords="offset points",
                    xytext=(-4, -12), ha="right", fontsize=7, color=COL["muted"])
        ax.annotate("Flan-T5 - 0.5", (ax.get_xlim()[0], flan_bacc - 0.5), textcoords="offset points",
                    xytext=(4, -10), fontsize=7, color=COL["muted"])
    ax.set_xlabel("simulated inference time per claim on a T4 (ms)", color=COL["muted"], fontsize=9)
    ax.set_ylabel("macro balanced accuracy (%)", color=COL["muted"], fontsize=9)
    ax.set_title(title, color=COL["text"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=COL["text"])
    fig.tight_layout()
    fig.savefig(path, facecolor=COL["surface"])
    plt.close(fig)


def difference_figure(summary):
    """Forest plot: macro BAcc difference vs Flan-T5 with paired-bootstrap 95% CI, and the -0.5 margin.
    This is the quantity the success criteria use. The oracle (+8.5) is left out so the scale stays readable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[~summary.approach.isin(["Flan-T5 alone", "Oracle deferral (reference)"])].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=200)
    fig.patch.set_facecolor(COL["surface"]); ax.set_facecolor(COL["surface"])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(COL["grid"])
    ax.grid(True, axis="x", color=COL["grid"], linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=COL["muted"], labelsize=8, left=False)
    ax.axvline(0, color=COL["muted"], lw=0.8, ls="--")
    ax.axvline(-0.5, color=COL["muted"], lw=0.8, ls=":")
    for i, r in enumerate(s.itertuples()):
        colour = (COL["selected"] if r.approach.startswith("Cascade") else
                  COL["random"] if r.approach.startswith("Random") else COL["base"])
        ax.plot([r.d_vs_flan_ci_low, r.d_vs_flan_ci_high], [i, i], color=colour, lw=2, solid_capstyle="round")
        ax.plot(r.d_bacc_vs_flan, i, "o", ms=7, color=colour, mec=COL["surface"], mew=2, zorder=5)
        ax.annotate(f"{r.d_bacc_vs_flan:+.2f} [{r.d_vs_flan_ci_low:+.2f}, {r.d_vs_flan_ci_high:+.2f}]",
                    (r.d_vs_flan_ci_high, i), textcoords="offset points", xytext=(6, -3), fontsize=7,
                    color=COL["muted"])
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([a.replace(" (same count per dataset, mean of draws)", "") for a in s.approach],
                       fontsize=8, color=COL["text"])
    lo = min(-3.5, s.d_vs_flan_ci_low.min() - 0.3)
    ax.set_xlim(lo, max(2.6, s.d_vs_flan_ci_high.max() + 1.4))
    ax.annotate("non-inferiority margin -0.5", (-0.5, len(s) - 0.4), xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=7, color=COL["muted"])
    ax.set_xlabel("test macro BAcc minus Flan-T5 (points), paired bootstrap 95% CI", color=COL["muted"], fontsize=9)
    ax.set_title("Test: difference from Flan-T5 alone", color=COL["text"], fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_test_vs_flan.png", facecolor=COL["surface"])
    plt.close(fig)


# ---------------------------------------------------------------- shared helpers
def curve_table(D):
    """Cascade BAcc, deferral rate and simulated cost for every delta on the grid (one row per delta)."""
    flan_total = D.lat_F.sum() / 1000
    rows = []
    for dv in deltas():
        pred, cost, d = cascade(D, dv)
        b, per = macro_bacc(D, pred)
        rows.append({"delta": dv, "deferred_pct": 100 * d.mean(), "ms_per_claim": cost.mean(),
                     "time_vs_flan": cost.sum() / 1000 / flan_total, "bacc": b, **per})
    return pd.DataFrame(rows)


def approaches(D, frozen, random_masks=None):
    """Every approach in a fixed order: name -> (pred, cost, deferred mask)."""
    out = {f"{NAMES[k]} alone": single(D, k) for k in ("R", "D", "F")}
    out["Majority vote (3 models)"] = ensemble(D)
    out[f"Cascade, primary (delta {frozen['primary']['delta']})"] = cascade(D, frozen["primary"]["delta"])
    if frozen["secondary"]["delta"] != frozen["primary"]["delta"]:
        out[f"Cascade, secondary (delta {frozen['secondary']['delta']})"] = cascade(D, frozen["secondary"]["delta"])
    out["Oracle deferral (reference)"] = oracle(D)
    return out


def md_table(df, floatfmt=2):
    """Render a DataFrame as a Markdown table; floats get `floatfmt` decimals and NaN becomes '-'."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---"
                                                          for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_baselines(D, split):
    """Recomputed single-model BAcc must equal the Phase 3 metrics files (baselines unchanged)."""
    summ = pd.read_csv(RESULTS / "metrics" / "summary.csv")
    for k, model in MODELS.items():
        ref = summ[(summ.model == model) & (summ.split == split)].bacc_macro.iloc[0]
        ours = macro_bacc(D, D[f"pred_{k}"].values)[0]
        assert abs(ours - ref) < 1e-3, f"{model}/{split}: {ours} != Phase 3 {ref}"


def dev_figure(D, curve, frozen):
    """Draw the development-split trade-off figure: delta curve, single models, majority vote and frozen deltas."""
    pts = [(NAMES[k], D[f"lat_{k}"].mean(), macro_bacc(D, D[f"pred_{k}"].values)[0], COL["base"], "o")
           for k in ("R", "D", "F")]
    ens = ensemble(D)
    pts.append(("Majority vote", ens[1].mean(), macro_bacc(D, ens[0])[0], COL["base"], "s"))
    for which in ("primary", "secondary"):
        f = frozen[which]
        if which == "primary" or f["delta"] != frozen["primary"]["delta"]:
            pts.append((f"{which}, delta {f['delta']}", f["dev_ms_per_claim"], f["dev_bacc"], COL["selected"], "D"))
    tradeoff_figure(OUT / "fig_dev_tradeoff.png", "Dev: accuracy vs simulated cost of the RoBERTa -> Flan-T5 "
                    "cascade (dashed: Flan-T5 alone)", curve, pts, D.lat_F.mean(), macro_bacc(D, D.pred_F.values)[0])


# ---------------------------------------------------------------- select (dev only)
def select(args):
    """Step 1, development split only: compute the delta curve, choose the primary and secondary
    delta with the prespecified rules and freeze them in results/phase4/selected_thresholds.json.

    Refuses to run if a delta is already frozen (unless --overwrite) or if test results exist.
    """
    if FROZEN.exists() and not args.overwrite:
        sys.exit(f"{FROZEN} exists: delta is already frozen. Re-selecting must be a deliberate, documented "
                 "decision (pass --overwrite and record why in PROGRESS.md).")
    if (OUT / "test_summary.csv").exists():
        sys.exit("test results already exist; selecting delta again after seeing test is not allowed")
    OUT.mkdir(parents=True, exist_ok=True)
    D = load("dev")
    check_baselines(D, "dev")
    curve = curve_table(D)
    curve.round(4).to_csv(OUT / "dev_delta_curve.csv", index=False)

    flan_bacc = macro_bacc(D, D.pred_F.values)[0]
    reach = curve[curve.bacc >= flan_bacc - 1e-9]
    best = curve.loc[curve.bacc.idxmax()]  # idxmax returns the first, i.e. the smallest delta on ties
    primary = reach.iloc[0] if len(reach) else best
    info = lambda r: {"delta": float(r.delta), "dev_bacc": round(float(r.bacc), 4),
                      "dev_deferred_pct": round(float(r.deferred_pct), 2),
                      "dev_ms_per_claim": round(float(r.ms_per_claim), 1),
                      "dev_time_vs_flan": round(float(r.time_vs_flan), 4)}
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_on": "dev",
        "rule_primary": P4["selection_on_dev"]["primary"],
        "rule_secondary": P4["selection_on_dev"]["secondary"],
        "flan_dev_bacc": round(flan_bacc, 4),
        "roberta_dev_bacc": round(macro_bacc(D, D.pred_R.values)[0], 4),
        "primary": {**info(primary), "reached_flan_on_dev": bool(len(reach))},
        "secondary": info(best),
        "phase4_config_sha256_16": sha(P4_FILE),
        "dev_inputs_sha256_16": {m: hashlib.sha256(b"".join((RESULTS / m / "dev" / f"{d}.csv").read_bytes()
                                                          for d in DATASETS)).hexdigest()[:16]
                                 for m in MODELS.values()},
    }
    FROZEN.write_text(json.dumps(frozen, indent=2))

    # Dev-level comparison of every approach at the frozen delta (no CIs; the test split carries the inference)
    flan_total = D.lat_F.sum() / 1000
    rows, per_ds = [], {}
    for name, (pred, cost, d) in approaches(D, frozen).items():
        r, per = describe(D, name, pred, cost, d, flan_total)
        rows.append(r); per_ds[name] = per
    pd.DataFrame(rows).round(3).to_csv(OUT / "dev_summary.csv", index=False)

    dev_figure(D, curve, frozen)
    print(json.dumps(frozen, indent=2))
    print(pd.DataFrame(rows)[["approach", "bacc", "deferred_pct", "ms_per_claim", "speedup_vs_flan"]]
          .round(3).to_string(index=False))


# ---------------------------------------------------------------- evaluate (test, once)
def evaluate(args):
    """Step 2, test split, run once: check that configs/phase4.json is unchanged since freezing, read
    the frozen delta, score every approach with paired-bootstrap CIs and the random-deferral
    control, and write results/phase4/ (the inputs of report Table 3).
    """
    if not FROZEN.exists():
        sys.exit("no frozen delta: run `phase4.py select` (dev) first")
    frozen = json.loads(FROZEN.read_text())
    if frozen["phase4_config_sha256_16"] != sha(P4_FILE):
        sys.exit("configs/phase4.json changed after delta was frozen")
    Ddev = load("dev")
    dev_figure(Ddev, pd.read_csv(OUT / "dev_delta_curve.csv"), frozen)
    D = load("test")
    check_baselines(D, "test")
    flan_total = D.lat_F.sum() / 1000
    apps = approaches(D, frozen)
    primary_name = next(n for n in apps if n.startswith("Cascade, primary"))
    p_pred, p_cost, p_def = apps[primary_name]

    # random deferral: same number of deferred claims per dataset as the primary cascade
    rcfg = P4["random_deferral_control"]
    n_def = {name: int(p_def[D.dataset.values == name].sum()) for name in DATASETS}
    masks = random_deferral(D, n_def, rcfg["draws"], rcfg["seed"])
    cR, cF = (D.pred_R == D.label).values, (D.pred_F == D.label).values
    rand_bacc, rand_cost, rand_per = [], [], []
    for mk in masks:
        b_, per_ = macro_bacc(D, np.where(mk, D.pred_F, D.pred_R))
        rand_bacc.append(b_); rand_per.append(per_)
        rand_cost.append((D.lat_R.values + mk * D.lat_F.values).sum() / 1000)
    rand_bacc, rand_cost = np.array(rand_bacc), np.array(rand_cost)
    q = D.dataset.map({name: n_def[name] / (D.dataset == name).sum() for name in DATASETS}).values
    rand_expected_correct = cR + q * (cF.astype(float) - cR)
    rand_expected_cost = D.lat_R.values + q * D.lat_F.values

    rows, per_ds = [], {}
    for name, (pred, cost, d) in apps.items():
        r, per = describe(D, name, pred, cost, d, flan_total)
        rows.append(r); per_ds[name] = per
    rname = "Random deferral (same count per dataset, mean of draws)"
    rr = {"approach": rname, "bacc": rand_bacc.mean(), "deferred_pct": 100 * p_def.mean(),
          "ms_per_claim": rand_expected_cost.mean(), "total_seconds": rand_expected_cost.sum() / 1000,
          "speedup_vs_flan": flan_total / (rand_expected_cost.sum() / 1000),
          "time_vs_flan": rand_expected_cost.sum() / 1000 / flan_total}
    rows.insert(len(rows) - 1, rr)

    # paired bootstrap
    b = P4["statistics"]
    correct = {n: (pred == D.label.values).astype(float) for n, (pred, _, _) in apps.items()}
    correct[rname] = rand_expected_correct
    costs = {n: cost for n, (_, cost, _) in apps.items()}
    costs[rname] = rand_expected_cost
    boot, boot_cost = bootstrap(D, correct, costs, b["replicates"], b["seed"])
    flan_name, rob_name = "Flan-T5 alone", "RoBERTa alone"
    for r in rows:
        r["bacc_ci_low"], r["bacc_ci_high"] = ci(boot[r["approach"]])
        dv = boot[r["approach"]] - boot[flan_name]
        r["d_bacc_vs_flan"] = r["bacc"] - next(x["bacc"] for x in rows if x["approach"] == flan_name)
        r["d_vs_flan_ci_low"], r["d_vs_flan_ci_high"] = ci(dv) if r["approach"] != flan_name else (np.nan, np.nan)
        sp = boot_cost[flan_name] / boot_cost[r["approach"]]
        r["speedup_ci_low"], r["speedup_ci_high"] = ci(sp) if r["approach"] != flan_name else (np.nan, np.nan)

    comparisons = {}
    for a, c in [(primary_name, flan_name), (primary_name, rob_name), (primary_name, rname),
                 ("Majority vote (3 models)", flan_name)] + \
                ([(n, flan_name) for n in apps if n.startswith("Cascade, secondary")]):
        diff = boot[a] - boot[c]
        point = next(x["bacc"] for x in rows if x["approach"] == a) - next(x["bacc"] for x in rows if x["approach"] == c)
        lo, hi = ci(diff)
        comparisons[f"{a} - {c}"] = {"diff": round(point, 3), "ci95": [round(lo, 3), round(hi, 3)],
                                     "share_of_replicates_le_0": round(float((diff <= 0).mean()), 4)}

    prow = next(r for r in rows if r["approach"] == primary_name)
    lo_vs_flan = comparisons[f"{primary_name} - {flan_name}"]["ci95"][0]
    verdict = {
        "outcome_1_better_than_flan": bool(lo_vs_flan > 0),
        "outcome_2_non_inferior": bool(lo_vs_flan > -0.5),
        "outcome_2_speedup_ok": bool(prow["time_vs_flan"] <= 0.8),
        "primary_time_vs_flan": round(prow["time_vs_flan"], 4),
        "primary_speedup": round(prow["speedup_vs_flan"], 3),
    }
    verdict["outcome_2_comparable_and_faster"] = verdict["outcome_2_non_inferior"] and verdict["outcome_2_speedup_ok"]
    verdict["result"] = ("positive: better than Flan-T5" if verdict["outcome_1_better_than_flan"] else
                         "positive: comparable to Flan-T5 and >= 1.25x faster" if verdict["outcome_2_comparable_and_faster"]
                         else "negative: neither success criterion met")
    random_summary = {"draws": int(rcfg["draws"]), "bacc_mean": round(float(rand_bacc.mean()), 3),
                      "bacc_2.5_97.5": [round(float(x), 3) for x in np.percentile(rand_bacc, [2.5, 97.5])],
                      "share_of_draws_ge_cascade": round(float((rand_bacc >= prow["bacc"]).mean()), 4),
                      "total_seconds_mean": round(float(rand_cost.mean()), 1),
                      "deferred_per_dataset": n_def}

    summary = pd.DataFrame(rows)
    summary.round(3).to_csv(OUT / "test_summary.csv", index=False)
    per = pd.DataFrame(per_ds).loc[DATASETS]
    per[rname] = pd.DataFrame(rand_per).mean()  # mean over the random draws
    per.loc["Average"] = per.mean()
    per.round(3).to_csv(OUT / "test_per_dataset.csv")
    (OUT / "test_statistics.json").write_text(json.dumps(
        {"frozen": frozen, "comparisons": comparisons, "verdict": verdict, "random_deferral": random_summary,
         "bootstrap": b, "note": "cost and speed-up are simulated from Phase 3 per-claim latencies"}, indent=2))

    test_curve = curve_table(D)
    test_curve.round(4).to_csv(OUT / "test_delta_curve_after_freezing.csv", index=False)
    errors = error_analysis(D, frozen, apps, primary_name)

    # figure
    pts = [(NAMES[k], D[f"lat_{k}"].mean(), macro_bacc(D, D[f"pred_{k}"].values)[0], COL["base"], "o")
           for k in ("R", "D", "F")]
    ens = apps["Majority vote (3 models)"]
    pts.append(("Majority vote", ens[1].mean(), macro_bacc(D, ens[0])[0], COL["base"], "s"))
    pts.append(("random deferral (mean)", rand_expected_cost.mean(), rand_bacc.mean(), COL["random"], "^"))
    for n, (pred, cost, _) in apps.items():
        if n.startswith("Cascade"):
            off = (7, 4, "left") if "primary" in n else (-8, 8, "right")
            pts.append((n.replace("Cascade, ", ""), cost.mean(), macro_bacc(D, pred)[0], COL["selected"], "D", off))
    tradeoff_figure(OUT / "fig_test_tradeoff.png",
                    "Test: frozen operating points, point estimates (dashed: Flan-T5 alone)",
                    None, pts, D.lat_F.mean(), macro_bacc(D, D.pred_F.values)[0],
                    extra=(test_curve, "test curve, computed after freezing (not used for selection)"))
    difference_figure(summary)
    write_report(summary, per, comparisons, verdict, random_summary, frozen, errors)
    sys.stdout.reconfigure(encoding="utf-8")
    print((OUT / "results_test.md").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- error analysis of the cascade
def error_analysis(D, frozen, apps, primary_name):
    """Where the deferral signal helps or fails (test, primary delta), plus the same confidence bins on dev."""
    delta = frozen["primary"]["delta"]
    out = {}
    # 1. RoBERTa confidence bins: is |p - 0.5| informative, and where is Flan-T5 better?
    bins = np.round(np.arange(0, 0.55, 0.05), 2)
    for split, DD in (("dev", load("dev")), ("test", D)):
        conf = (DD.p_R - 0.5).abs()
        b = pd.cut(conf, bins, right=False, include_lowest=True)
        cR, cF = DD.pred_R == DD.label, DD.pred_F == DD.label
        t = pd.DataFrame({"bin": b, "cR": cR, "cF": cF, "fix": ~cR & cF, "break": cR & ~cF, "latF": DD.lat_F})
        g = t.groupby("bin", observed=False).agg(n=("cR", "size"), roberta_acc=("cR", "mean"), flan_acc=("cF", "mean"),
                                                 flan_fixes=("fix", "sum"), flan_breaks=("break", "sum"),
                                                 flan_ms=("latF", "mean"))
        g["share_pct"] = 100 * g.n / g.n.sum()
        g[["roberta_acc", "flan_acc"]] *= 100
        g.index = [f"[{iv.left:.2f}, {iv.right:.2f})" for iv in g.index]
        out[f"confidence_bins_{split}"] = g.reset_index(names="|p_RoBERTa - 0.5|")
    # 2. outcome of each deferral decision (test)
    d = deferred(D, delta)
    cR, cF = (D.pred_R == D.label).values, (D.pred_F == D.label).values
    out["deferral_outcomes"] = pd.DataFrame([
        {"group": "deferred: RoBERTa wrong, Flan-T5 right (fixed)", "n": int((d & ~cR & cF).sum())},
        {"group": "deferred: RoBERTa right, Flan-T5 wrong (broken)", "n": int((d & cR & ~cF).sum())},
        {"group": "deferred: both right", "n": int((d & cR & cF).sum())},
        {"group": "deferred: both wrong", "n": int((d & ~cR & ~cF).sum())},
        {"group": "kept: RoBERTa right", "n": int((~d & cR).sum())},
        {"group": "kept: RoBERTa wrong, Flan-T5 would be right (missed)", "n": int((~d & ~cR & cF).sum())},
        {"group": "kept: RoBERTa wrong, Flan-T5 also wrong", "n": int((~d & ~cR & ~cF).sum())},
    ])
    out["deferral_outcomes"]["pct_of_claims"] = 100 * out["deferral_outcomes"].n / len(D)
    # 3. per dataset
    pred = apps[primary_name][0]
    rows = []
    for name in DATASETS:
        m = D.dataset.values == name
        sub = D[m]
        bR = macro_bacc_ds(sub, sub.pred_R.values); bF = macro_bacc_ds(sub, sub.pred_F.values)
        bC = macro_bacc_ds(sub, pred[m])
        rows.append({"dataset": name, "claims": int(m.sum()), "deferred_pct": 100 * d[m].mean(),
                     "bacc_roberta": bR, "bacc_flan": bF, "bacc_cascade": bC, "cascade_minus_flan": bC - bF,
                     "time_vs_flan": (D.lat_R.values[m] + d[m] * D.lat_F.values[m]).sum() / D.lat_F.values[m].sum()})
    out["per_dataset"] = pd.DataFrame(rows)
    # 4. cost by document length (RoBERTa chunks)
    lb = pd.cut(D.chunks_R, [0, 1, 3, 7, 10**6], labels=["1 chunk", "2-3", "4-7", "8+"])
    cost = D.lat_R.values + d * D.lat_F.values
    t = pd.DataFrame({"len": lb, "d": d, "cost": cost, "flan": D.lat_F.values})
    g = t.groupby("len", observed=False).agg(n=("d", "size"), deferred_pct=("d", "mean"), cascade_s=("cost", "sum"),
                                             flan_s=("flan", "sum"))
    g["deferred_pct"] *= 100
    g["share_of_claims_pct"] = 100 * g.n / g.n.sum()
    g["share_of_cascade_time_pct"] = 100 * g.cascade_s / g.cascade_s.sum()
    g[["cascade_s", "flan_s"]] /= 1000
    out["cost_by_length"] = g.reset_index(names="document length (RoBERTa chunks)")
    # 5. error direction, all approaches
    rows = []
    for n, (p, _, _) in apps.items():
        y = D.label.values
        rows.append({"approach": n, "missed_errors_FP": int(((p == 1) & (y == 0)).sum()),
                     "false_alarms_FN": int(((p == 0) & (y == 1)).sum())})
    out["error_direction"] = pd.DataFrame(rows)
    for k, v in out.items():
        v.round(3).to_csv(OUT / f"error_{k}.csv", index=False)
    return out


def macro_bacc_ds(sub, pred):
    """Balanced accuracy (%) of `pred` on one dataset's rows `sub` (mean of the two class recalls)."""
    y, c = sub.label.values, pred == sub.label.values
    return 50 * (c[y == 1].mean() + c[y == 0].mean())


def write_report(summary, per, comparisons, verdict, random_summary, frozen, errors):
    """Write results/phase4/results_test.md: main table, precision/recall/F1, per-dataset BAcc and verdict."""
    s = summary.copy()
    s["BAcc (95% CI)"] = [f"{b:.2f} [{l:.2f}, {h:.2f}]" for b, l, h in zip(s.bacc, s.bacc_ci_low, s.bacc_ci_high)]
    s["Δ vs Flan-T5 (95% CI)"] = ["–" if np.isnan(l) else f"{d:+.2f} [{l:+.2f}, {h:+.2f}]"
                                  for d, l, h in zip(s.d_bacc_vs_flan, s.d_vs_flan_ci_low, s.d_vs_flan_ci_high)]
    main = s[["approach", "BAcc (95% CI)", "Δ vs Flan-T5 (95% CI)", "deferred_pct", "ms_per_claim",
              "total_seconds", "speedup_vs_flan"]].rename(columns={
                  "deferred_pct": "% deferred", "ms_per_claim": "ms/claim (sim.)",
                  "total_seconds": "total s (sim.)", "speedup_vs_flan": "speed-up vs Flan-T5 (sim.)"})
    prf = s[["approach", "precision_supported", "recall_supported", "f1_supported", "precision_unsupported",
             "recall_unsupported", "f1_unsupported", "macro_f1", "tp", "fn", "fp", "tn"]].dropna(subset=["macro_f1"])
    lines = [
        "# Phase 4 results on the test split (simulated cascade)",
        "",
        f"Frozen on dev before this evaluation (`selected_thresholds.json`, {frozen['frozen_at']}): primary delta = "
        f"{frozen['primary']['delta']}, secondary delta = {frozen['secondary']['delta']}.",
        "Times are **simulated**: per-claim latencies measured separately for each model on the Kaggle T4 in Phase 3.",
        "",
        f"**Verdict: {verdict['result']}.**",
        "",
        "## All approaches",
        "", md_table(main), "",
        "BAcc = mean of the 10 per-dataset balanced accuracies. CIs: paired bootstrap, 10,000 resamples within datasets.",
        "",
        "## Precision, recall, F1 (macro over datasets; positive class = supported) and pooled confusion counts",
        "", md_table(prf), "",
        "## Paired bootstrap comparisons", "",
        md_table(pd.DataFrame([{"comparison": k, "diff": v["diff"], "CI low": v["ci95"][0], "CI high": v["ci95"][1],
                                "share of replicates <= 0": v["share_of_replicates_le_0"]}
                               for k, v in comparisons.items()]), 3), "",
        f"Random deferral: {random_summary['draws']} draws, BAcc mean {random_summary['bacc_mean']}, "
        f"95% of draws in {random_summary['bacc_2.5_97.5']}; share of draws >= cascade: "
        f"{random_summary['share_of_draws_ge_cascade']}.",
        "",
        "## Success criteria (fixed in configs/phase4.json)", "",
        "```", json.dumps(verdict, indent=2), "```", "",
        "## BAcc per dataset", "", md_table(per.reset_index(names="dataset")), "",
        "## Error analysis of the primary cascade", "",
    ]
    for k, v in errors.items():
        lines += [f"### {k.replace('_', ' ')}", "", md_table(v), ""]
    (OUT / "results_test.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- measured (after the Kaggle cascade run)
def measured(args):
    """Compare the end-to-end Kaggle cascade run with the simulation (same frozen delta, same test claims)."""
    frozen = json.loads(FROZEN.read_text())
    run = args.results / "cascade" / "test"
    missing = [d for d in DATASETS if not (run / f"{d}.csv").exists()]
    if missing:
        sys.exit(f"cascade run incomplete, missing {missing}")
    M = pd.concat([pd.read_csv(run / f"{d}.csv") for d in DATASETS]).set_index("idx").sort_index()
    metas = [json.loads((run / f"{d}.json").read_text()) for d in DATASETS]
    runs = [json.loads(l) for l in (args.results / "cascade" / "runs.jsonl").read_text().splitlines()]
    if any(abs(m["delta"] - frozen["primary"]["delta"]) > 1e-12 for m in metas):
        sys.exit("the Kaggle run used a different delta from the frozen one")
    D = load("test")
    if not (M.index.equals(D.index) and (M.label.values == D.label.values).all()):
        sys.exit("measured cascade ids/labels differ from the Phase 3 test predictions")
    sim_pred, sim_cost, sim_def = cascade(D, frozen["primary"]["delta"])
    dm = M.deferred.values.astype(bool)
    flan_total = D.lat_F.sum() / 1000
    total = M.latency_ms.sum() / 1000
    res = {
        "delta": frozen["primary"]["delta"], "n": len(M),
        "deferral_decisions_equal_to_simulation": int((dm == sim_def).sum()),
        "predictions_equal_to_simulation": int((M.pred.values == sim_pred).sum()),
        "max_abs_diff_prob_roberta_vs_phase3": float((M.prob_r - D.p_R).abs().max()),
        "max_abs_diff_prob_flan_vs_phase3_on_deferred": float((M.prob_f[dm] - D.p_F[dm]).abs().max()) if dm.any() else None,
        "bacc_measured": macro_bacc(D, M.pred.values.astype(int))[0], "bacc_simulated": macro_bacc(D, sim_pred)[0],
        "deferred_pct_measured": 100 * dm.mean(), "deferred_pct_simulated": 100 * sim_def.mean(),
        "end_to_end_seconds_measured": total,
        "stage_sum_seconds_measured": (M.latency_r_ms.sum() + M.latency_f_ms.fillna(0).sum()) / 1000,
        "seconds_simulated": sim_cost.sum() / 1000,
        "ms_per_claim_measured": M.latency_ms.mean(), "ms_per_claim_simulated": sim_cost.mean(),
        "flan_alone_seconds_phase3": flan_total, "roberta_alone_seconds_phase3": D.lat_R.sum() / 1000,
        "speedup_vs_flan_measured": flan_total / total, "speedup_vs_flan_simulated": flan_total / (sim_cost.sum() / 1000),
        "roberta_stage_time_vs_phase3": M.latency_r_ms.sum() / D.lat_R.sum(),
        "flan_stage_time_vs_phase3_same_claims": M.latency_f_ms[dm].sum() / D.lat_F[dm].sum() if dm.any() else None,
        "peak_gpu_mb_allocated": max(m.get("peak_gpu_mb_allocated", 0) for m in metas),
        "peak_gpu_mb_reserved": max(m.get("peak_gpu_mb_reserved", 0) for m in metas),
        "weights_gpu_mb_both_models": runs[-1].get("weights_gpu_mb"),
        "load_seconds_both_models": runs[-1].get("load_seconds"),
        "batch_fallbacks_roberta": sum(m.get("n_batch_fallbacks_roberta-large", 0) for m in metas),
        "batch_fallbacks_flan": sum(m.get("n_batch_fallbacks_flan-t5-large", 0) for m in metas),
        "oom_retry_seconds_included_in_end_to_end": sum(m.get("oom_retry_seconds", 0) for m in metas),
        "gpu": sorted({m["env"]["gpu"] for m in metas}),
        "note": "end-to-end = whole claim incl. both stages, the decision and OOM retries; "
                "simulated = Phase 3 per-claim latencies combined; Flan-T5 alone = Phase 3 (another session)",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measured_vs_simulated.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


def main():
    """Command-line entry point: `python scripts/phase4.py {select|evaluate|measured}`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["select", "evaluate", "measured"])
    ap.add_argument("--overwrite", action="store_true", help="select: replace an existing frozen delta (document why)")
    ap.add_argument("--results", type=Path, default=RESULTS, help="measured: folder holding cascade/test from Kaggle")
    args = ap.parse_args()
    {"select": select, "evaluate": evaluate, "measured": measured}[args.step](args)


if __name__ == "__main__":
    main()
