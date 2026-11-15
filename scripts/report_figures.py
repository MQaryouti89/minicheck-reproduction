"""Redraw the report's figures at print size (16 cm wide, 9-10 pt text) from saved result files only.

No model is run and no statistic is recomputed: every value is read from results/final, results/phase4 or
results/phase5. The figures in results/final/fig*.png are unchanged; these are the same data, drawn larger
and without in-figure titles (the report captions carry them).

Usage: python scripts/report_figures.py      -> results/final/report_figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIN, P4, P5 = ROOT / "results" / "final", ROOT / "results" / "phase4", ROOT / "results" / "phase5"
OUT = FIN / "report_figures"
COL = {"text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "surface": "#ffffff",
       "roberta": "#2a78d6", "flan": "#eb6834", "deberta": "#1baf7a", "random": "#1baf7a", "base": "#52514e"}
TICK, LABEL, NOTE = 9, 9.5, 8.5
W = 6.3  # inches = 16 cm, the report's text width

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def setup(h, grid="both"):
    fig, ax = plt.subplots(figsize=(W, h), dpi=300)
    fig.patch.set_facecolor(COL["surface"]); ax.set_facecolor(COL["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["grid"])
    ax.tick_params(colors=COL["muted"], labelsize=TICK)
    if grid:
        ax.grid(True, axis=grid, color=COL["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def xlabel(ax, s):
    ax.set_xlabel(s, color=COL["muted"], fontsize=LABEL)


def ylabel(ax, s):
    ax.set_ylabel(s, color=COL["muted"], fontsize=LABEL)


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=NOTE, labelcolor=COL["text"], **kw)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=COL["surface"])
    plt.close(fig)


def note(ax, text, xy, dx=6, dy=4, ha="left", colour=None, size=NOTE, box=False):
    ax.annotate(text, xy, textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=size,
                color=colour or COL["text"], zorder=7,
                bbox=dict(fc=COL["surface"], ec="none", pad=0.6) if box else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_csv(FIN / "T1_reproduction_vs_paper.csv")
    t3 = pd.read_csv(FIN / "T3_all_approaches_test.csv")
    # accuracy and CIs from the unrounded values (scripts/unrounded_ci.py), so 2-decimal labels are exact
    exact = pd.read_csv(FIN / "T3_unrounded.csv").set_index("approach")
    base = t3.approach.str.replace(", end to end", "", regex=False)
    for c in exact.columns:
        t3[c] = exact.loc[base, c].values
    frozen = json.loads((P4 / "selected_thresholds.json").read_text())
    meas = json.loads((P4 / "measured_vs_simulated.json").read_text())

    # Figure 1: development split, delta selection
    curve = pd.read_csv(P4 / "dev_delta_curve.csv")
    dev = pd.read_csv(P4 / "dev_summary.csv").set_index("approach")
    fig, ax = setup(3.5)
    ax.axhline(frozen["flan_dev_bacc"], color=COL["muted"], lw=0.8, ls="--")
    ax.plot(curve.ms_per_claim, curve.bacc, color=COL["roberta"], lw=1.8, label="cascade, δ from 0 to 0.5")
    for name, mk in [("RoBERTa alone", "o"), ("DeBERTa alone", "o"), ("Flan-T5 alone", "o"), ("Majority vote (3 models)", "s")]:
        x, y = dev.loc[name, "ms_per_claim"], dev.loc[name, "bacc"]
        ax.plot(x, y, mk, ms=7, color=COL["base"], mec=COL["surface"], mew=1.5, zorder=5)
        lab = name.replace(" alone", "").replace(" (3 models)", "")
        dx, dy, ha = {"RoBERTa": (8, -14, "left"), "Flan-T5": (6, -15, "left"), "Majority vote": (-6, 7, "right")}.get(lab, (8, 5, "left"))
        note(ax, lab, (x, y), dx=dx, dy=dy, ha=ha)
    for key, lab, off in [("primary", "primary δ = 0.19", (45, -45)), ("secondary", "secondary δ = 0.41", (50, 4))]:
        r = frozen[key]
        ax.plot(r["dev_ms_per_claim"], r["dev_bacc"], "D", ms=8, color=COL["flan"], mec=COL["surface"], mew=1.5, zorder=6)
        ax.annotate(lab, (r["dev_ms_per_claim"], r["dev_bacc"]), textcoords="offset points", xytext=off, fontsize=NOTE,
                    color=COL["text"], va="center", arrowprops=dict(arrowstyle="-", color=COL["muted"], lw=0.7))
    ax.set_ylim(70.85, 73.65)
    xlabel(ax, "simulated inference time per claim on a T4 (ms)")
    ylabel(ax, "dev macro BAcc (%)")
    legend(ax, loc="lower right")
    save(fig, "fig1_dev_delta_selection.png")

    # Figure 2: difference from Flan-T5 with paired-bootstrap CIs (oracle left out for scale)
    s = t3[~t3.approach.isin(["Flan-T5 alone", "Oracle deferral (reference)"]) & ~t3.approach.str.contains("end to end")]
    s = s.iloc[::-1].reset_index(drop=True)
    names = {"Cascade, primary (delta 0.19)": "Cascade δ = 0.19", "Cascade, secondary (delta 0.41)": "Cascade δ = 0.41",
             "Random deferral (same count per dataset, mean of draws)": "Random deferral", "Majority vote (3 models)": "Majority vote"}
    fig, ax = setup(3.0, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.axvline(0, color=COL["muted"], lw=0.8, ls="--")
    ax.axvline(-0.5, color=COL["muted"], lw=0.9, ls=":")
    for i, r in s.iterrows():
        c = COL["flan"] if r.approach.startswith("Cascade") else COL["random"] if r.approach.startswith("Random") else COL["base"]
        ax.plot([r.d_vs_flan_ci_low, r.d_vs_flan_ci_high], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(r.d_bacc_vs_flan, i, "o", ms=7, color=c, mec=COL["surface"], mew=1.5, zorder=5)
        note(ax, f"{r.d_bacc_vs_flan:+.2f} [{r.d_vs_flan_ci_low:+.2f}, {r.d_vs_flan_ci_high:+.2f}]".replace("-", "−"),
             (1.75, i), dx=0, dy=-3, colour=COL["muted"])  # one aligned column, clear of the lines
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([names.get(a, a) for a in s.approach], fontsize=TICK, color=COL["text"])
    ax.set_xlim(-3.6, 4.1)
    ax.set_xticks(range(-3, 2))
    ax.set_ylim(-0.6, len(s) - 0.2)
    note(ax, "margin −0.5", (-0.5, len(s) - 0.55), dx=-4, dy=0, ha="right", colour=COL["muted"])
    xlabel(ax, "test macro BAcc minus Flan-T5 (points), 95% CI")
    save(fig, "fig2_difference_vs_flan_ci.png")

    # Figure 3: accuracy vs cost (test)
    fig, ax = setup(3.6)
    flan = t3[t3.approach == "Flan-T5 alone"].iloc[0]
    ax.axhline(flan.bacc, color=COL["muted"], lw=0.8, ls="--")
    for _, r in t3.iterrows():
        a = r.approach
        if a.startswith("Oracle"):
            continue
        c = COL["flan"] if a.startswith("Cascade") else COL["random"] if a.startswith("Random") else COL["base"]
        ax.plot([r.ms_per_claim] * 2, [r.bacc_ci_low, r.bacc_ci_high], color=c, lw=1.2, alpha=0.4)
        if a.startswith("Cascade"):
            hollow = "end to end" in a
            lab = ("cascade δ = 0.19, measured end to end" if hollow else
                   "cascade δ = 0.19, simulated" if "primary" in a else "cascade δ = 0.41, simulated")
            ax.plot(r.ms_per_claim, r.bacc, "D" if "primary" in a else "d", ms=8, mew=2, zorder=5, label=lab,
                    color=c, mfc=COL["surface"] if hollow else c, mec=c if hollow else COL["surface"])
            continue
        if a.startswith("Random"):
            ax.plot(r.ms_per_claim, r.bacc, "^", ms=8, color=c, mec=COL["surface"], mew=1.5, zorder=5,
                    label="random deferral, same counts (mean)")
            continue
        mk = "s" if a.startswith("Majority") else "o"
        ax.plot(r.ms_per_claim, r.bacc, mk, ms=7, color=c, mec=COL["surface"], mew=1.5, zorder=5)
        lab = a.replace(" alone", "").replace(" (3 models)", "")
        dx, ha = (-8, "right") if lab in ("Majority vote", "RoBERTa") else (8, "left")
        note(ax, lab, (r.ms_per_claim, r.bacc), dx=dx, dy=4, ha=ha)
    ax.set_xlim(90, 900)
    xlabel(ax, "inference time per claim on a Kaggle T4 (ms); bars = 95% CI")
    ylabel(ax, "test macro BAcc (%)")
    legend(ax, loc="lower right")
    save(fig, "fig3_accuracy_vs_cost.png")

    # Figure 4: accuracy by RoBERTa confidence
    bins = pd.read_csv(P4 / "error_confidence_bins_test.csv")
    mids = np.arange(len(bins)) * 0.05 + 0.025
    delta = frozen["primary"]["delta"]
    fig, ax = setup(3.3, grid="y")
    ax.axvspan(0, delta, color=COL["roberta"], alpha=0.08, lw=0)
    ax.text(delta / 2, 59.6, f"deferred (δ = {delta})", ha="center", fontsize=NOTE, color=COL["muted"])
    ax.plot(mids, bins.roberta_acc, "o-", color=COL["roberta"], lw=2, ms=5, label="RoBERTa")
    ax.plot(mids, bins.flan_acc, "s-", color=COL["flan"], lw=2, ms=5, label="Flan-T5")
    for x, sh in zip(mids, bins.share_pct):
        ax.annotate(f"{sh:.0f}%", (x, 43.5), ha="center", fontsize=NOTE, color=COL["muted"])
    ax.set_ylim(42, 86)
    xlabel(ax, "RoBERTa confidence |p − 0.5| (bins of 0.05); % = share of claims")
    ylabel(ax, "accuracy in the bin (%)")
    legend(ax, loc="upper left")
    save(fig, "fig4_confidence_bins.png")

    # Figure 5: claim types within datasets
    wd = pd.read_csv(P5 / "claim_types_within_dataset.csv")
    fig, ax = setup(3.2, grid="x")
    ax.axvline(0, color=COL["muted"], lw=0.8)
    ticks = []
    for i, feat in enumerate(["numeric", "negation", "complex"]):
        sub = wd[wd["claim type"] == feat]
        for j, (col, c, lab) in enumerate([("Flan-T5 - RoBERTa, with flag", COL["flan"], "claims with the flag"),
                                           ("Flan-T5 - RoBERTa, without", COL["base"], "claims without the flag")]):
            yv = 2 * (2 - i) + (0.35 if j == 0 else -0.35)
            ax.plot(sub[col], [yv] * len(sub), "o", ms=5, color=c, alpha=0.45, mec="none")
            ax.plot(sub[col].mean(), yv, "D", ms=8, color=c, mec=COL["surface"], mew=1.5,
                    label=lab if i == 0 else None)
        ticks.append((2 * (2 - i), f"{feat}\n({len(sub)} datasets)"))
    ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([l for _, l in ticks], fontsize=TICK, color=COL["text"])
    ax.tick_params(left=False); ax.spines["left"].set_visible(False)
    xlabel(ax, "Flan-T5 BAcc minus RoBERTa BAcc within a dataset (points)")
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig5_claim_types.png")

    # Figure C1: our BAcc minus the paper's, per dataset
    d = t1.iloc[::-1].reset_index(drop=True)
    fig, ax = setup(3.6, grid="x")
    ax.axvspan(-0.05, 0.05, color=COL["grid"], alpha=0.7, lw=0)
    ax.axvline(0, color=COL["muted"], lw=0.8)
    for model, lab, c, off in [("roberta-large", "RoBERTa", COL["roberta"], 0.22), ("deberta-v3-large", "DeBERTa", COL["deberta"], 0),
                               ("flan-t5-large", "Flan-T5", COL["flan"], -0.22)]:
        ax.plot(d[f"{model} diff"], np.arange(len(d)) + off, "o", ms=6, color=c, mec=COL["surface"], mew=1, label=lab)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.dataset, fontsize=TICK, color=COL["text"])
    ax.tick_params(left=False)
    xlabel(ax, "our test BAcc minus the paper's Table 2 (points)")
    legend(ax, loc="lower right")
    save(fig, "figC1_reproduction_vs_paper.png")

    # Figure C2: who is wrong, per dataset
    a = pd.read_csv(P5 / "agreement_by_dataset.csv")
    a = a[a.dataset != "All"].iloc[::-1].reset_index(drop=True)
    fig, ax = setup(3.6, grid="x")
    left = np.zeros(len(a))
    for col, c, lab in [("only RoBERTa wrong %", COL["roberta"], "only RoBERTa wrong"),
                        ("only Flan-T5 wrong %", COL["flan"], "only Flan-T5 wrong"),
                        ("both wrong %", COL["base"], "both wrong")]:
        ax.barh(range(len(a)), a[col], left=left, color=c, height=0.62, label=lab, edgecolor=COL["surface"], lw=1)
        left += a[col].values
    for i, v in enumerate(left):
        note(ax, f"{v:.0f}%", (v, i), dx=4, dy=-3, colour=COL["muted"])
    ax.set_yticks(range(len(a))); ax.set_yticklabels(a.dataset, fontsize=TICK, color=COL["text"])
    ax.tick_params(left=False); ax.set_xlim(0, 75)
    xlabel(ax, "% of the dataset's test claims")
    legend(ax, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3)
    save(fig, "figC2_error_categories.png")

    # Figure C3: document length vs share of time
    g = pd.read_csv(P5 / "length_and_cost.csv")
    fig, ax = setup(3.0, grid="y")
    x, w = np.arange(len(g)), 0.38
    ax.bar(x - w / 2, g["share of claims %"], w, color=COL["base"], label="share of test claims")
    ax.bar(x + w / 2, g["share of cascade time %"], w, color=COL["roberta"], label="share of cascade time (simulated)")
    for xi, (s1, s2) in enumerate(zip(g["share of claims %"], g["share of cascade time %"])):
        note(ax, f"{s1:.0f}%", (xi - w / 2, s1), dx=0, dy=2, ha="center", colour=COL["muted"])
        note(ax, f"{s2:.0f}%", (xi + w / 2, s2), dx=0, dy=2, ha="center", colour=COL["muted"])
    ax.set_xticks(x); ax.set_xticklabels(g.iloc[:, 0], fontsize=TICK, color=COL["text"])
    xlabel(ax, "document length (number of RoBERTa chunks)")
    ylabel(ax, "%")
    legend(ax, loc="upper right")
    save(fig, "figC3_length_cost.png")

    # Figure C4: total time by stage (simulated vs measured)
    r_alone, f_alone = meas["roberta_alone_seconds_phase3"], meas["flan_alone_seconds_phase3"]
    sim_f = meas["seconds_simulated"] - r_alone
    m_r = r_alone * meas["roberta_stage_time_vs_phase3"]
    m_f = meas["stage_sum_seconds_measured"] - m_r
    bars = [("Flan-T5 alone", 0, f_alone), ("RoBERTa alone", r_alone, 0), ("Cascade, simulated", r_alone, sim_f),
            ("Cascade, measured", m_r, m_f)][::-1]
    fig, ax = setup(2.4, grid="x")
    for i, (lab, rs, fs) in enumerate(bars):
        ax.barh(i, rs, color=COL["roberta"], height=0.55, label="RoBERTa stage" if i == 0 else None)
        ax.barh(i, fs, left=rs, color=COL["flan"], height=0.55, label="Flan-T5 stage" if i == 0 else None,
                edgecolor=COL["surface"], lw=1)
        note(ax, f"{rs + fs:,.0f} s", (rs + fs, i), dx=4, dy=-3, colour=COL["muted"])
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars], fontsize=TICK, color=COL["text"])
    ax.tick_params(left=False); ax.set_xlim(0, 6000)
    xlabel(ax, "total inference time on the 12,949 test claims (s)")
    legend(ax, loc="lower right")
    save(fig, "figC4_simulated_vs_measured.png")
    print("report figures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
