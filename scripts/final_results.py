"""Final tables, figures and consistency checks for the report (CPU only, reads saved results only).

    python scripts/final_results.py

Writes results/final/: tables.md (+ one CSV per table), fig_*.png, consistency_check.md, and
manifest_sha256.txt (hashes of every result file, so later changes are detectable).
Every number is recomputed from the per-claim prediction files where possible and compared with the saved
Phase 3 / 4 / 5 outputs and with README.md. The script exits with an error if anything disagrees.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import phase4  # noqa: E402
from phase4 import COL, DATASETS, MODELS, NAMES, macro_bacc  # noqa: E402

R3 = ROOT / "results" / "kaggle-t4" / "minicheck_results"
P4 = ROOT / "results" / "phase4"
P5 = ROOT / "results" / "phase5"
CASCADE_RUN = ROOT / "results" / "kaggle-t4-cascade" / "minicheck_results"
OUT = ROOT / "results" / "final"
CFG = phase4.CFG
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})


def fig_setup(w=7.5, h=4.2):
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


MODEL_COL = {"roberta-large": COL["cascade"], "deberta-v3-large": COL["random"], "flan-t5-large": COL["selected"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    per = pd.read_csv(R3 / "metrics" / "per_dataset.csv")
    summ = pd.read_csv(R3 / "metrics" / "summary.csv")
    t4 = pd.read_csv(P4 / "test_summary.csv")
    stats = json.loads((P4 / "test_statistics.json").read_text())
    meas = json.loads((P4 / "measured_vs_simulated.json").read_text())
    frozen = json.loads((P4 / "selected_thresholds.json").read_text())

    # ---- recompute from raw predictions and compare with saved files
    data = {s: phase4.load(s) for s in ("dev", "test")}
    for split, D in data.items():
        for k, model in MODELS.items():
            m, perds = macro_bacc(D, D[f"pred_{k}"].values)
            saved = per[(per.model == model) & (per.split == split) & per.dataset.isin(DATASETS)].set_index("dataset").bacc
            ok = all(abs(perds[d] - saved[d]) < 1e-3 for d in DATASETS)
            check(f"{model} {split}: per-dataset BAcc = metrics/per_dataset.csv", ok)
            ref = summ[(summ.model == model) & (summ.split == split)].bacc_macro.iloc[0]
            check(f"{model} {split}: macro BAcc {m:.2f} = metrics/summary.csv", abs(m - ref) < 1e-3, f"{m:.3f} vs {ref:.3f}")
            if split == "test":
                ref4 = t4[t4.approach == f"{NAMES[k]} alone"].bacc.iloc[0]
                check(f"{model} test: macro BAcc = phase4/test_summary.csv", abs(m - ref4) < 1e-3)
                tot = D[f"lat_{k}"].sum() / 1000
                ref_t = summ[(summ.model == model) & (summ.split == "test")].inference_seconds.iloc[0]
                check(f"{model} test: total time {tot:.0f} s = metrics/summary.csv", abs(tot - ref_t) < 0.1)
    D = data["test"]
    delta = frozen["primary"]["delta"]
    pred_c, cost_c, d = phase4.cascade(D, delta)
    b_c = macro_bacc(D, pred_c)[0]
    prow = t4[t4.approach.str.startswith("Cascade, primary")].iloc[0]
    check("cascade: BAcc recomputed = phase4/test_summary.csv", abs(b_c - prow.bacc) < 1e-3, f"{b_c:.3f}")
    check("cascade: BAcc measured on Kaggle = simulated", abs(meas["bacc_measured"] - b_c) < 1e-9)
    check("cascade: deferral % recomputed = test_summary = measured",
          abs(100 * d.mean() - prow.deferred_pct) < 1e-3 and abs(meas["deferred_pct_measured"] - 100 * d.mean()) < 1e-9,
          f"{100 * d.mean():.2f}%")
    check("cascade: all 12,949 measured predictions equal the simulation",
          meas["predictions_equal_to_simulation"] == meas["n"] == 12949)
    check("frozen delta file unchanged (sha256 ce2102326bc8c686)",
          hashlib.sha256((P4 / "selected_thresholds.json").read_bytes()).hexdigest().startswith("ce2102326bc8c686"))
    p5 = pd.read_csv(P5 / "per_dataset.csv").set_index("dataset")
    p4d = pd.read_csv(P4 / "test_per_dataset.csv", index_col=0)
    pc = [c for c in p4d.columns if c.startswith("Cascade, primary")][0]
    check("phase5 per-dataset cascade BAcc = phase4/test_per_dataset.csv",
          all(abs(p5.loc[x, "Cascade"] - p4d.loc[x, pc]) < 1e-3 for x in DATASETS))
    # measured run: recompute from the raw Kaggle files
    run = pd.concat([pd.read_csv(CASCADE_RUN / "cascade" / "test" / f"{x}.csv") for x in DATASETS])
    check("measured: end-to-end seconds recomputed from cascade/test/*.csv",
          abs(run.latency_ms.sum() / 1000 - meas["end_to_end_seconds_measured"]) < 0.01,
          f"{run.latency_ms.sum() / 1000:.1f} s")

    # ---- T1 reproduction vs paper (test) and T2 dev+test, cost
    rows = []
    for x in DATASETS + ["Average"]:
        r = {"dataset": x}
        for model in MODELS.values():
            paper = dict(zip(DATASETS, CFG["paper_table2"][model]["bacc"]))
            ours = per[(per.model == model) & (per.split == "test") & (per.dataset == x)].bacc.iloc[0]
            pap = np.mean(CFG["paper_table2"][model]["bacc"]) if x == "Average" else paper[x]
            r[f"{model} paper"] = pap
            r[f"{model} ours"] = ours
            r[f"{model} diff"] = ours - pap
        rows.append(r)
    tables["T1_reproduction_vs_paper"] = pd.DataFrame(rows)
    rows = []
    for model in MODELS.values():
        dv = summ[(summ.model == model) & (summ.split == "dev")].iloc[0]
        ts = summ[(summ.model == model) & (summ.split == "test")].iloc[0]
        rows.append({"model": model, "dev BAcc": dv.bacc_macro, "test BAcc": ts.bacc_macro,
                     "paper (row mean / printed)": f"{ts.paper_row_mean:.2f} / {ts.paper_printed_average}",
                     "datasets matching paper": ts.datasets_matching_paper, "test macro F1": ts.macro_f1_mean,
                     "ms/claim (test)": ts.ms_per_claim, "dev + test time (min)": (dv.inference_seconds + ts.inference_seconds) / 60,
                     "load (s)": ts.load_seconds, "weights GPU (MiB)": ts.weights_gpu_mb,
                     "peak GPU (MiB)": max(dv.peak_gpu_mb_allocated, ts.peak_gpu_mb_allocated),
                     "OOM fallback claims": int(dv.batch_fallbacks + ts.batch_fallbacks)})
    tables["T2_baselines_dev_test_cost"] = pd.DataFrame(rows)

    # ---- T3 all approaches (test), with the measured cascade added
    t3 = t4[["approach", "bacc", "bacc_ci_low", "bacc_ci_high", "d_bacc_vs_flan", "d_vs_flan_ci_low",
             "d_vs_flan_ci_high", "deferred_pct", "ms_per_claim", "total_seconds", "speedup_vs_flan"]].copy()
    t3.insert(1, "cost", "simulated")
    t3.loc[t3.approach.str.endswith("alone"), "cost"] = "measured (Phase 3)"
    t3.loc[t3.approach.str.startswith("Majority"), "cost"] = "sum of measured"
    mrow = {"approach": f"Cascade, primary (delta {delta}), end to end", "cost": "measured (Kaggle run)",
            "bacc": meas["bacc_measured"], "bacc_ci_low": prow.bacc_ci_low, "bacc_ci_high": prow.bacc_ci_high,
            "d_bacc_vs_flan": prow.d_bacc_vs_flan, "d_vs_flan_ci_low": prow.d_vs_flan_ci_low,
            "d_vs_flan_ci_high": prow.d_vs_flan_ci_high, "deferred_pct": meas["deferred_pct_measured"],
            "ms_per_claim": meas["ms_per_claim_measured"], "total_seconds": meas["end_to_end_seconds_measured"],
            "speedup_vs_flan": meas["speedup_vs_flan_measured"]}
    pos = int(np.flatnonzero(t3.approach.str.startswith("Cascade, primary"))[0]) + 1
    t3 = pd.concat([t3.iloc[:pos], pd.DataFrame([mrow]), t3.iloc[pos:]], ignore_index=True)
    # Δ vs Flan-T5 = BAcc - Flan-T5 BAcc, from the unrounded cascade BAcc where available; the Phase 4 CSV keeps only
    # 3 decimals, and rounding its -0.205 again would give -0.20 instead of the exact -0.2052 -> -0.21
    flan_exact = macro_bacc(D, D.pred_F.values)[0]
    t3.loc[t3.approach.str.startswith("Cascade, primary"), "bacc"] = b_c
    t3["d_bacc_vs_flan"] = t3.bacc - flan_exact
    t3.loc[t3.approach == "Flan-T5 alone", ["bacc", "d_bacc_vs_flan"]] = [flan_exact, 0.0]
    check("cascade: Δ vs Flan-T5 = −0.2052 (exact) and matches the bootstrap point difference",
          abs(b_c - flan_exact - stats["comparisons"][next(k for k in stats["comparisons"] if k.startswith("Cascade, primary")
                                                             and k.endswith("Flan-T5 alone"))]["diff"]) < 1e-3,
          f"{b_c - flan_exact:.4f}")
    tables["T3_all_approaches_test"] = t3
    tables["T4_paired_bootstrap"] = pd.DataFrame(
        [{"comparison": k, "difference": v["diff"], "CI low": v["ci95"][0], "CI high": v["ci95"][1],
          "share of replicates <= 0": v["share_of_replicates_le_0"]} for k, v in stats["comparisons"].items()]
        + [{"comparison": "random deferral, 1,000 draws: BAcc mean and 2.5-97.5% range",
            "difference": stats["random_deferral"]["bacc_mean"], "CI low": stats["random_deferral"]["bacc_2.5_97.5"][0],
            "CI high": stats["random_deferral"]["bacc_2.5_97.5"][1],
            "share of replicates <= 0": None}])
    tables["T5_simulated_vs_measured"] = pd.DataFrame([
        {"quantity": "predictions equal to simulation", "simulated": "-", "measured": f"{meas['predictions_equal_to_simulation']:,} / {meas['n']:,}"},
        {"quantity": "test macro BAcc", "simulated": f"{b_c:.2f}", "measured": f"{meas['bacc_measured']:.2f}"},
        {"quantity": "% sent to Flan-T5", "simulated": f"{100 * d.mean():.1f}", "measured": f"{meas['deferred_pct_measured']:.1f}"},
        {"quantity": "total inference time (s)", "simulated": f"{meas['seconds_simulated']:,.0f}", "measured": f"{meas['end_to_end_seconds_measured']:,.0f}"},
        {"quantity": "ms per claim", "simulated": f"{meas['ms_per_claim_simulated']:.0f}", "measured": f"{meas['ms_per_claim_measured']:.0f}"},
        {"quantity": "speed-up vs Flan-T5 alone", "simulated": f"{meas['speedup_vs_flan_simulated']:.2f}x", "measured": f"{meas['speedup_vs_flan_measured']:.2f}x"},
        {"quantity": "RoBERTa stage time vs Phase 3", "simulated": "1.00", "measured": f"{meas['roberta_stage_time_vs_phase3']:.2f}"},
        {"quantity": "Flan-T5 stage time vs Phase 3 (same claims)", "simulated": "1.00", "measured": f"{meas['flan_stage_time_vs_phase3_same_claims']:.2f}"},
        {"quantity": "GPU memory of both models' weights (MiB)", "simulated": "-", "measured": f"{meas['weights_gpu_mb_both_models']:,.0f}"},
        {"quantity": "peak GPU memory allocated (MiB)", "simulated": "-", "measured": f"{meas['peak_gpu_mb_allocated']:,.0f}"},
        {"quantity": "out-of-memory fallbacks", "simulated": "-", "measured": str(meas["batch_fallbacks_roberta"] + meas["batch_fallbacks_flan"])},
    ])
    tables["T6_precision_recall_f1_test"] = t4[["approach", "precision_supported", "recall_supported", "f1_supported",
                                                "precision_unsupported", "recall_unsupported", "f1_unsupported",
                                                "macro_f1", "tp", "fn", "fp", "tn"]].dropna(subset=["macro_f1"])
    ce = pd.read_csv(P5 / "confident_errors.csv")
    ag = pd.read_csv(P5 / "agreement_by_dataset.csv").set_index("dataset").loc["All"]
    lc = pd.read_csv(P5 / "length_and_cost.csv")
    r_err = int(ce.roberta_errors.sum())
    kept = int(ce.roberta_errors.iloc[1:].sum())
    fixable = int(ce.flan_would_fix.sum())
    tables["T7_error_analysis_key_numbers"] = pd.DataFrame([
        {"finding": "RoBERTa errors (test)", "value": f"{r_err:,}"},
        {"finding": "... of which kept by the gate (|p-0.5| >= delta)", "value": f"{kept:,} ({100 * kept / r_err:.1f}%)"},
        {"finding": "RoBERTa errors that Flan-T5 gets right", "value": f"{fixable:,}; deferred: {int(ce.flan_would_fix.iloc[0])} ({100 * ce.flan_would_fix.iloc[0] / fixable:.1f}%)"},
        {"finding": "claims where both RoBERTa and Flan-T5 are wrong", "value": f"{ag['both wrong %']:.1f}%"},
        {"finding": "claims where all three checkers are wrong", "value": f"{ag['all three wrong %']:.1f}%"},
        {"finding": "ExpertQA: claims where all three are wrong", "value": f"{pd.read_csv(P5 / 'agreement_by_dataset.csv').set_index('dataset').loc['ExpertQA', 'all three wrong %']:.1f}%"},
        {"finding": "RoBERTa errors in its most confident bin (|p-0.5| >= 0.45)", "value": f"{int(ce.roberta_errors.iloc[-1]):,} ({ce['share of all RoBERTa errors %'].iloc[-1]:.1f}%)"},
        {"finding": "8+ chunk documents: share of claims / of cascade time", "value": f"{lc['share of claims %'].iloc[-1]:.1f}% / {lc['share of cascade time %'].iloc[-1]:.1f}%"},
    ])
    # ---- claim types (Phase 5, needs the gated text once; outputs hold flags and ids only)
    if (P5 / "claim_types.csv").exists():
        ct = pd.read_csv(P5 / "claim_types.csv")
        ws = pd.read_csv(P5 / "claim_types_within_dataset_summary.csv")
        tables["T8_claim_types_pooled"] = ct[["claim type", "present", "claims", "share %", "RoBERTa BAcc", "RoBERTa CI",
                                              "Flan-T5 BAcc", "Flan-T5 CI", "Cascade BAcc", "Cascade CI",
                                              "Flan-T5 - RoBERTa BAcc", "Flan-T5 - RoBERTa CI", "deferred %"]]
        tables["T9_claim_types_within_dataset"] = ws
        feats = pd.read_csv(P5 / "claim_features.csv")
        check("claim features: one row per test claim, same ids as the predictions",
              len(feats) == len(D) and set(feats.idx) == set(D.index))
        check("claim features / examples: no claim or document text columns",
              not ({"claim", "doc"} & set(feats.columns)) and
              not ({"claim", "doc"} & set(pd.read_csv(P5 / "examples_idx.csv").columns)))
        ex = pd.read_csv(P5 / "examples_idx.csv").set_index("idx")
        check("failure examples: ids, labels and datasets match the predictions",
              ex.index.isin(D.index).all() and (ex.label == D.loc[ex.index, "label"]).all()
              and (ex.dataset == D.loc[ex.index, "dataset"]).all())
        pooled_c = ct[(ct["claim type"] == "complex") & ct.present]["Cascade BAcc"].iloc[0]
        cm = pd.read_csv(P5 / "claim_features.csv").set_index("idx").loc[D.index, "complex"].values
        yy = D.label.values
        rec = 50 * ((pred_c == yy)[cm & (yy == 1)].mean() + (pred_c == yy)[cm & (yy == 0)].mean())
        check("claim types: complex-claim cascade BAcc recomputed from flags + predictions",
              abs(rec - pooled_c) < 1e-3, f"{rec:.2f}")

        # F10: Flan-T5 minus RoBERTa within each dataset, claims with vs without the flag
        wd = pd.read_csv(P5 / "claim_types_within_dataset.csv")
        fig, ax, plt = fig_setup(7.5, 3.6)
        ax.grid(True, axis="x", color=COL["grid"], linewidth=0.6)
        ax.axvline(0, color=COL["muted"], lw=0.8)
        ylabels = []
        for i, feat in enumerate(["numeric", "negation", "complex"]):
            sub = wd[wd["claim type"] == feat]
            for j, (col, colour, lab) in enumerate([("Flan-T5 - RoBERTa, with flag", COL["selected"], "claims with the flag"),
                                                    ("Flan-T5 - RoBERTa, without", COL["base"], "claims without")]):
                yv = 2 * (2 - i) + (0.4 if j == 0 else -0.4)
                ax.plot(sub[col], [yv] * len(sub), "o", ms=4, color=colour, alpha=0.45, mec="none")
                ax.plot(sub[col].mean(), yv, "D", ms=8, color=colour, mec=COL["surface"], mew=1.5,
                        label=f"{lab} (diamond = mean)" if i == 0 else None)
            ylabels.append((2 * (2 - i), feat + "\n" + f"({len(sub)} datasets)"))
        ax.set_yticks([y for y, _ in ylabels]); ax.set_yticklabels([l for _, l in ylabels], fontsize=8, color=COL["text"])
        ax.tick_params(left=False)
        ax.set_xlabel("Flan-T5 BAcc minus RoBERTa BAcc within a dataset (points); dots = datasets",
                      color=COL["muted"], fontsize=8.5)
        ax.set_title("Where Flan-T5 helps: complex claims (within-dataset comparison)", color=COL["text"],
                     fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=COL["text"])
        fig.tight_layout(); fig.savefig(OUT / "fig10_claim_types.png", facecolor=COL["surface"]); plt.close(fig)

    for k, v in tables.items():
        v.round(4).to_csv(OUT / f"{k}.csv", index=False)

    # ---- figures
    # F1: our BAcc minus the paper, per dataset
    fig, ax, plt = fig_setup(7.5, 4.4)
    ax.grid(True, axis="x", color=COL["grid"], linewidth=0.6)
    ax.axvspan(-0.05, 0.05, color=COL["grid"], alpha=0.6, lw=0)
    t1 = tables["T1_reproduction_vs_paper"].iloc[::-1].reset_index(drop=True)
    for j, (model, off) in enumerate(zip(MODELS.values(), (-0.22, 0, 0.22))):
        ax.plot(t1[f"{model} diff"], np.arange(len(t1)) + off, "o", ms=6, color=MODEL_COL[model],
                mec=COL["surface"], mew=1.5, label=NAMES[[k for k, v in MODELS.items() if v == model][0]])
    ax.set_yticks(range(len(t1))); ax.set_yticklabels(t1.dataset, fontsize=8, color=COL["text"])
    ax.axvline(0, color=COL["muted"], lw=0.8)
    ax.set_xlabel("our test BAcc minus the paper's Table 2 (points); grey band = within the paper's rounding (±0.05)",
                  color=COL["muted"], fontsize=8)
    ax.set_title("Reproduction: all three checkers match the paper", color=COL["text"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=COL["text"])
    fig.tight_layout(); fig.savefig(OUT / "fig1_reproduction_vs_paper.png", facecolor=COL["surface"]); plt.close(fig)

    # F2: per-dataset BAcc of the three models
    fig, ax, plt = fig_setup(7.5, 4.4)
    ax.grid(True, axis="x", color=COL["grid"], linewidth=0.6)
    t1f = tables["T1_reproduction_vs_paper"]
    t1f = t1f[t1f.dataset != "Average"].iloc[::-1].reset_index(drop=True)
    for model in MODELS.values():
        k = [k for k, v in MODELS.items() if v == model][0]
        ax.plot(t1f[f"{model} ours"], range(len(t1f)), "o", ms=7, color=MODEL_COL[model], mec=COL["surface"],
                mew=1.5, label=NAMES[k])
    for i in range(len(t1f)):
        vals = [t1f.loc[i, f"{m} ours"] for m in MODELS.values()]
        ax.plot([min(vals), max(vals)], [i, i], color=COL["grid"], lw=2, zorder=0)
    ax.set_yticks(range(len(t1f))); ax.set_yticklabels(t1f.dataset, fontsize=8, color=COL["text"])
    ax.set_xlabel("test balanced accuracy (%)", color=COL["muted"], fontsize=8.5)
    ax.set_title("Per-dataset BAcc of the three released checkers (Kaggle T4)", color=COL["text"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=COL["text"])
    fig.tight_layout(); fig.savefig(OUT / "fig2_per_dataset_bacc.png", facecolor=COL["surface"]); plt.close(fig)

    # F3: accuracy vs cost (test)
    fig, ax, plt = fig_setup(7.5, 4.6)
    ax.grid(True, color=COL["grid"], linewidth=0.6)
    pts = []
    for _, r in t3.iterrows():
        a = r.approach
        if a.startswith("Oracle"):
            continue
        colour = (COL["selected"] if a.startswith("Cascade") else COL["random"] if a.startswith("Random") else COL["base"])
        marker = "D" if a.startswith("Cascade") else "^" if a.startswith("Random") else "s" if a.startswith("Majority") else "o"
        label = (a.replace(" alone", "").replace(" (same count per dataset, mean of draws)", " (mean)")
                 .replace("Cascade, ", "cascade ").replace(" (3 models)", ""))
        pts.append((label, r.ms_per_claim, r.bacc, colour, marker, r.bacc_ci_low, r.bacc_ci_high))
    for lab, x, y, colour, marker, lo, hi in pts:
        ax.plot([x, x], [lo, hi], color=colour, lw=1, alpha=0.35)
        if lab.startswith("cascade"):  # three close points: legend instead of direct labels
            hollow = "end to end" in lab
            name = ("cascade δ 0.19, measured end to end" if hollow else
                    "cascade δ 0.19, simulated" if "primary" in lab else "cascade δ 0.41 (secondary), simulated")
            ax.plot(x, y, "D" if "primary" in lab else "d", ms=8, color=colour, zorder=5, mew=2, label=name,
                    mfc=COL["surface"] if hollow else colour, mec=colour if hollow else COL["surface"])
            continue
        ax.plot(x, y, marker, ms=8, color=colour, zorder=5, mew=2, mec=COL["surface"])
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=7.5, color=COL["text"])
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=COL["text"])
    ax.axhline(t3[t3.approach == "Flan-T5 alone"].bacc.iloc[0], color=COL["muted"], lw=0.8, ls="--")
    ax.set_xlabel("inference time per claim on a Kaggle T4 (ms); vertical lines = 95% CI of BAcc",
                  color=COL["muted"], fontsize=8.5)
    ax.set_ylabel("test macro balanced accuracy (%)", color=COL["muted"], fontsize=8.5)
    ax.set_title("Accuracy vs inference cost (dashed: Flan-T5 alone)", color=COL["text"], fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig3_accuracy_vs_cost.png", facecolor=COL["surface"]); plt.close(fig)

    # F5: simulated vs measured time
    fig, ax, plt = fig_setup(7.5, 3.0)
    ax.grid(True, axis="x", color=COL["grid"], linewidth=0.6)
    sim_r, sim_f = D.lat_R.sum() / 1000, (d * D.lat_F.values).sum() / 1000
    m_r = run.latency_r_ms.sum() / 1000
    m_f = run.latency_f_ms.fillna(0).sum() / 1000
    bars = [("Flan-T5 alone (Phase 3)", 0, D.lat_F.sum() / 1000), ("RoBERTa alone (Phase 3)", D.lat_R.sum() / 1000, 0),
            ("Cascade, simulated", sim_r, sim_f), ("Cascade, measured end to end", m_r, m_f)][::-1]
    for i, (lab, r_s, f_s) in enumerate(bars):
        ax.barh(i, r_s, color=COL["cascade"], height=0.55, label="RoBERTa stage" if i == 0 else None)
        ax.barh(i, f_s, left=r_s, color=COL["selected"], height=0.55, label="Flan-T5 stage" if i == 0 else None,
                edgecolor=COL["surface"], lw=1)
        ax.annotate(f"{r_s + f_s:,.0f} s", (r_s + f_s, i), xytext=(4, -3), textcoords="offset points", fontsize=7.5,
                    color=COL["muted"])
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars], fontsize=8, color=COL["text"])
    ax.tick_params(left=False)
    ax.set_xlim(0, 5800)
    ax.set_xlabel("total inference time on the 12,949 test claims (s)", color=COL["muted"], fontsize=8.5)
    ax.set_title("Simulated vs measured cascade time", color=COL["text"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=COL["text"])
    fig.tight_layout(); fig.savefig(OUT / "fig5_simulated_vs_measured.png", facecolor=COL["surface"]); plt.close(fig)
    check("figure 5: measured stage sum = measured json",
          abs(m_r + m_f - meas["stage_sum_seconds_measured"]) < 0.01)

    # copies of Phase 4/5 figures, so results/final is complete
    for src, dst in [(P4 / "fig_test_vs_flan.png", "fig4_difference_vs_flan_ci.png"),
                     (P4 / "fig_dev_tradeoff.png", "fig6_dev_delta_selection.png"),
                     (P5 / "fig_confidence_bins.png", "fig7_confidence_bins.png"),
                     (P5 / "fig_error_categories.png", "fig8_error_categories.png"),
                     (P5 / "fig_length_cost.png", "fig9_length_cost.png")]:
        shutil.copy2(src, OUT / dst)

    # ---- README consistency
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    must = ["73.14", "72.56", "74.71", "74.51", "−0.21 [−1.22, +0.82]", "1.47×", "3,380", "13.3 GiB", "13.6",
            "+1.12 [+0.50, +1.75]", "+1.37 [+0.67, +2.10]", "0.19", "0.41", "12,949 / 12,949"]
    missing = [s for s in must if s not in readme]
    check("README.md contains the key final numbers", not missing, f"missing: {missing}" if missing else "")

    # ---- write tables.md, consistency report, manifest
    lines = ["# Final tables", "", "Generated by `scripts/final_results.py` from the saved results. "
             "BAcc in %, macro-averaged over the 10 datasets unless stated.", ""]
    for k, v in tables.items():
        lines += [f"## {k.replace('_', ' ')}", "", phase4.md_table(v.round(3), 2), ""]
    (OUT / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    cdf = pd.DataFrame(checks)
    (OUT / "consistency_check.md").write_text("# Consistency check\n\n" + phase4.md_table(cdf) + "\n", encoding="utf-8")
    files = sorted(p for p in (ROOT / "results").rglob("*") if p.is_file() and "final" not in p.parts
                   and p.suffix != ".lnk")  # .lnk: Windows shortcut, git-ignored
    manifest = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n" for p in files)
    (OUT / "manifest_sha256.txt").write_bytes(manifest.encode())  # LF line endings: `sha256sum -c` works anywhere
    sys.stdout.reconfigure(encoding="utf-8")
    print(cdf.to_string(index=False))
    if (cdf.result == "FAIL").any():
        sys.exit("consistency check FAILED")


if __name__ == "__main__":
    main()
