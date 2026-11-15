"""Unrounded test BAcc, differences and paired-bootstrap CIs, for correct 2-decimal reporting.

results/phase4/test_summary.csv (and hence results/final/T3) stores CI bounds rounded to 3 decimals, and
rounding those again to 2 decimals can differ from rounding the exact value (e.g. the random-deferral lower
bound is stored as -2.365 but is -2.36497, i.e. -2.36). This script repeats `phase4.py evaluate` in memory,
with the same frozen file, configuration and seeds, writing into a temporary folder. It then checks that the
number files it produced are byte-identical to results/phase4, and saves only the unrounded values.
Nothing in results/phase4 is modified, and no threshold or setting is chosen again.

Usage: python scripts/unrounded_ci.py      -> results/final/T3_unrounded.csv
"""
import contextlib
import filecmp
import io
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import phase4  # noqa: E402

COLS = ["approach", "bacc", "bacc_ci_low", "bacc_ci_high", "d_bacc_vs_flan", "d_vs_flan_ci_low", "d_vs_flan_ci_high"]
COMPARE = ["test_summary.csv", "test_per_dataset.csv", "test_statistics.json", "test_delta_curve_after_freezing.csv"]


def main():
    src = ROOT / "results" / "phase4"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in ["selected_thresholds.json", "dev_delta_curve.csv"]:
            shutil.copy(src / f, tmp / f)
        phase4.OUT, phase4.FROZEN = tmp, tmp / "selected_thresholds.json"
        captured = {}
        draw = phase4.difference_figure

        def grab(summary):
            captured["summary"] = summary.copy()
            draw(summary)

        phase4.difference_figure = grab
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                phase4.evaluate(None)
            except AttributeError:  # the final print reconfigures stdout, which StringIO cannot do
                pass
        same = {f: filecmp.cmp(tmp / f, src / f, shallow=False) for f in COMPARE}
    if not all(same.values()):
        sys.exit(f"re-run differs from results/phase4: {same}; not writing")
    s = captured["summary"][COLS]
    s.to_csv(ROOT / "results" / "final" / "T3_unrounded.csv", index=False, float_format="%.6f")
    print("identical to results/phase4:", ", ".join(same))
    print(s.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
