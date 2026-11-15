"""Tests for scripts/error_analysis.py on synthetic claims (no gated data). Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import error_analysis as ea  # noqa: E402
import phase4  # noqa: E402


@pytest.mark.parametrize("claim, numeric, negation, complex_", [
    ("The bridge opened in 1932.", True, False, False),
    ("Revenue rose by two percent.", True, False, False),
    ("The minister did not attend the meeting.", False, True, False),
    ("She isn't the author.", False, True, False),
    ("He never met the president and the vice president, who was abroad.", False, True, True),
    ("The company, which was founded by two engineers, sold phones and tablets.", True, False, True),
    ("Cats sleep a lot.", False, False, False),
    (" ".join(["word"] * 26), False, False, True),
    ("Nothing was announced.", False, True, False),
    ("Notable people attended.", False, False, False),  # 'not' inside a word is not a negation
])
def test_claim_features(claim, numeric, negation, complex_):
    """The claim-type word rules give the expected flags on hand-written claims."""
    f = ea.claim_features(claim)
    assert (f["numeric"], f["negation"], f["complex"]) == (numeric, negation, complex_)


def synthetic(n_per=30, seed=0):
    """Synthetic predictions and claim texts shaped like the real files (no gated data)."""
    rng = np.random.default_rng(seed)
    rows, texts = [], []
    k = 0
    for ds in phase4.DATASETS:
        for _ in range(n_per):
            y = int(rng.integers(0, 2))
            p = {m: float(np.clip(rng.normal(0.5 + 0.2 * (2 * y - 1), 0.25), 0.001, 0.999)) for m in "RDF"}
            rows.append({"idx": f"test-{k:05d}", "dataset": ds, "label": y,
                         **{f"p_{m}": p[m] for m in "RDF"}, **{f"pred_{m}": int(p[m] > 0.5) for m in "RDF"},
                         **{f"lat_{m}": 50.0 for m in "RDF"}, **{f"chunks_{m}": 1 for m in "RDF"}})
            claim = rng.choice(["Prices did not rise in 2020.", "The team won and the coach, who was new, stayed.",
                                "The river is long.", "Three ministers resigned."])
            texts.append({"idx": f"test-{k:05d}", "dataset": ds, "label": y, "doc": "Short document. " * 5,
                          "claim": claim})
            k += 1
    D = pd.DataFrame(rows).set_index("idx")
    T = pd.DataFrame(texts).set_index("idx")
    return D, T


def test_text_outputs_contain_no_text(tmp_path, monkeypatch):
    """Claim-type outputs written to results/ contain no claim or document text."""
    monkeypatch.setattr(ea, "OUT", tmp_path / "phase5")
    monkeypatch.setattr(ea, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(ea, "REPS", 50)
    (tmp_path / "phase5").mkdir()
    D, T = synthetic()
    pred_c, cost, d = phase4.cascade(D, 0.19)
    F, types, prev = ea.text_analyses(D, T, pred_c, d, 0.19)
    assert set(types["claim type"]) == {"numeric", "negation", "complex"}
    assert np.isclose(types.groupby("claim type").claims.sum(), 2 * len(D) / 2).all()  # present + absent = all
    ex = ea.select_examples(D, T, F, pred_c, d)
    for f in (tmp_path / "phase5").glob("*.csv"):
        content = f.read_text()
        assert "Prices did not rise" not in content and "Short document" not in content, f.name
    assert "Short document" in (tmp_path / "private" / "phase5_examples.md").read_text(encoding="utf-8")
    assert ex.groupby("category").dataset.apply(lambda s: s.is_unique).all()  # one per dataset per category


def test_text_must_match_predictions():
    """Text analyses stop if the text file's labels do not match the predictions."""
    D, T = synthetic()
    T.loc[T.index[0], "label"] = 1 - T.loc[T.index[0], "label"]
    pred_c, _, d = phase4.cascade(D, 0.19)
    with pytest.raises(AssertionError):
        ea.text_analyses(D, T, pred_c, d, 0.19)


def test_within_dataset_and_short_documents():
    """Within-dataset comparisons and the short-document check handle normal and empty cases."""
    D, T = synthetic(n_per=60)
    pred_c, _, d = phase4.cascade(D, 0.19)
    F = pd.DataFrame([ea.claim_features(c) for c in T.claim], index=D.index)
    per, summ = ea.within_dataset(D, F, pred_c, min_per_class=3)
    assert set(summ["claim type"].dropna()) <= {"numeric", "negation", "complex"}
    assert (per["claims with flag"] + per["claims without"] == 60).all()  # both groups of one dataset
    empty, _ = ea.within_dataset(D, F, pred_c, min_per_class=10**6)       # nothing eligible: no crash
    assert empty.empty
    short = ea.short_documents(D, T, limit=10**6)                          # every document counts as short
    assert short.set_index("dataset").loc["All", "docs < 1000000 chars"] == len(D)
