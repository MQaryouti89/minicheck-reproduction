"""Draw the cascade diagram of the report (Fig. 1) from saved result files.

The frozen delta comes from results/phase4/selected_thresholds.json, the deferral rate of the primary
cascade from results/phase4/test_summary.csv, and the per-claim test latencies of RoBERTa and Flan-T5
from results/kaggle-t4/minicheck_results/metrics/summary.csv. No model is run.

Usage: python scripts/cascade_diagram.py      -> results/final/report_figures/fig_cascade_diagram.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "final" / "report_figures"
COL = {"text": "#0b0b0b", "muted": "#52514e", "surface": "#ffffff", "roberta": "#2a78d6", "flan": "#eb6834"}
TITLE, SUB = 7.6, 6.8
plt.rcParams.update({"font.family": "DejaVu Sans"})


def inputs():
    """Read delta, the primary cascade's deferral rate (%) and the RoBERTa / Flan-T5 test ms per claim."""
    delta = json.loads((ROOT / "results/phase4/selected_thresholds.json").read_text())["primary"]["delta"]
    summ = pd.read_csv(ROOT / "results/phase4/test_summary.csv")
    deferred = float(summ.loc[summ.approach.str.startswith("Cascade, primary"), "deferred_pct"].iloc[0])
    m = pd.read_csv(ROOT / "results/kaggle-t4/minicheck_results/metrics/summary.csv")
    ms = {k: float(m.query("model == @k and split == 'test'").ms_per_claim.iloc[0])
          for k in ("roberta-large", "flan-t5-large")}
    return delta, deferred, ms


def box(ax, x, y, w, h, title, sub, edge, fill="#ffffff"):
    """Draw a rounded box with a bold title line and a smaller second line."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=1.2, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=TITLE, color=COL["text"], weight="bold")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=SUB, color=COL["muted"])


def arrow(ax, x1, y1, x2, y2, label=None, dx=0, dy=0):
    """Draw an arrow from (x1, y1) to (x2, y2), with an optional label offset by (dx, dy)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COL["muted"], shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, ha="center", va="center", fontsize=SUB, color=COL["text"])


def main():
    """Draw and save the diagram (16 cm wide, like the other report figures)."""
    delta, deferred, ms = inputs()
    fig, ax = plt.subplots(figsize=(6.3, 2.4), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 38); ax.axis("off")
    y0, h = 23, 10
    box(ax, 0.5, y0, 13, h, "Document", "+ claim", COL["muted"])
    box(ax, 18, y0, 22, h, "MiniCheck-RoBERTa", f"score pR ({ms['roberta-large']:.0f} ms/claim)", COL["roberta"], "#eaf2fc")
    arrow(ax, 14.1, y0 + h / 2, 17.6, y0 + h / 2)
    cx, cy, dw, dh = 55.5, y0 + h / 2, 11.5, 8
    ax.add_patch(Polygon([(cx - dw, cy), (cx, cy + dh), (cx + dw, cy), (cx, cy - dh)], closed=True,
                         linewidth=1.2, edgecolor=COL["text"], facecolor="#ffffff"))
    ax.text(cx, cy, "|pR − 0.5| < δ ?", ha="center", va="center", fontsize=TITLE, color=COL["text"], weight="bold")
    arrow(ax, 40.6, cy, cx - dw, cy)
    box(ax, 76, y0, 23.5, h, "Keep RoBERTa's label", "supported if pR > 0.5", COL["roberta"])
    arrow(ax, cx + dw, cy, 75.6, cy, f"no ({100 - deferred:.1f}%)", 0, 1.8)
    box(ax, 44.5, 2, 22, h, "MiniCheck-Flan-T5", f"score pF ({ms['flan-t5-large']:.0f} ms/claim)", COL["flan"], "#fdeee7")
    arrow(ax, cx, cy - dh, cx, 12.4, f"yes ({deferred:.1f}%)", 7.5, 0.2)
    box(ax, 76, 2, 23.5, h, "Use Flan-T5's label", "supported if pF > 0.5", COL["flan"])
    arrow(ax, 66.9, 7, 75.6, 7)
    ax.text(0.5, 7, f"δ = {delta}, selected on development\ndata and frozen before testing.\n"
            "Shares: 12,949 test claims;\nlatencies: Kaggle T4.",
            ha="left", va="center", fontsize=SUB, color=COL["muted"], linespacing=1.35)
    fig.tight_layout(pad=0.15)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_cascade_diagram.png", facecolor=COL["surface"])
    plt.close(fig)
    print("wrote", (OUT / "fig_cascade_diagram.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
