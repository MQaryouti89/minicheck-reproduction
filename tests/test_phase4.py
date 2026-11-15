"""Tests for scripts/phase4.py on synthetic predictions (no real data, no GPU). Run: python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import phase4  # noqa: E402


def fake_results(root, seed=0):
    """Phase 3-shaped prediction files with the real claim counts, for all three models and both splits."""
    rng = np.random.default_rng(seed)
    summary = []
    for split in ("dev", "test"):
        n_total = sum(phase4.CFG["expected_counts"][split].values())
        ids = [f"{split}-{i:05d}" for i in range(n_total)]
        ds = np.concatenate([[d] * phase4.CFG["expected_counts"][split][d] for d in phase4.DATASETS])
        y = rng.integers(0, 2, n_total)
        chunks = rng.integers(1, 8, n_total)
        for k, model in phase4.MODELS.items():
            skill = {"R": 1.0, "D": 0.9, "F": 1.4}[k]
            logit = skill * (2 * y - 1) + rng.normal(0, 1.3, n_total)
            prob = 1 / (1 + np.exp(-logit))
            lat = chunks * {"R": 40, "D": 60, "F": 90}[k] * rng.uniform(0.8, 1.2, n_total)
            df = pd.DataFrame({"idx": ids, "dataset": ds, "label": y, "prob": prob, "pred": (prob > 0.5).astype(int),
                               "n_chunks": chunks, "latency_ms": lat})
            for d in phase4.DATASETS:
                out = root / model / split
                out.mkdir(parents=True, exist_ok=True)
                df[df.dataset == d].to_csv(out / f"{d}.csv", index=False)
            per = [balanced_accuracy_score(y[ds == d], (prob > 0.5)[ds == d]) * 100 for d in phase4.DATASETS]
            summary.append({"model": model, "split": split, "bacc_macro": np.mean(per)})
    (root / "metrics").mkdir()
    pd.DataFrame(summary).to_csv(root / "metrics" / "summary.csv", index=False)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    res, out = tmp_path / "res", tmp_path / "phase4"
    fake_results(res)
    monkeypatch.setattr(phase4, "RESULTS", res)
    monkeypatch.setattr(phase4, "OUT", out)
    monkeypatch.setattr(phase4, "FROZEN", out / "selected_thresholds.json")
    small = json.loads(json.dumps(phase4.P4))
    small["statistics"]["replicates"] = 300
    small["random_deferral_control"]["draws"] = 50
    monkeypatch.setattr(phase4, "P4", small)
    return res, out


def test_cascade_endpoints_and_cost(env):
    D = phase4.load("dev")
    pred, cost, d = phase4.cascade(D, 0.0)
    assert not d.any() and (pred == D.pred_R.values).all() and np.allclose(cost, D.lat_R.values)
    pred, cost, d = phase4.cascade(D, phase4.deltas()[-1])
    assert d.all() and (pred == D.pred_F.values).all() and np.allclose(cost, D.lat_R.values + D.lat_F.values)


def test_macro_bacc_matches_sklearn(env):
    D = phase4.load("test")
    ours, per = phase4.macro_bacc(D, D.pred_D.values)
    ref = [100 * balanced_accuracy_score(D.label[D.dataset == d], D.pred_D[D.dataset == d]) for d in phase4.DATASETS]
    assert np.allclose(list(per.values()), ref) and np.isclose(ours, np.mean(ref))


def test_bootstrap_paired_and_expected_correctness(env):
    D = phase4.load("test")
    c = (D.pred_R == D.label).values.astype(float)
    boot, tot = phase4.bootstrap(D, {"a": c, "b": c}, {"a": D.lat_R.values}, replicates=200, seed=1)
    assert np.array_equal(boot["a"], boot["b"])  # same resamples for every approach
    assert abs(boot["a"].mean() - phase4.macro_bacc(D, D.pred_R.values)[0]) < 0.5
    # BAcc is linear in correctness, so expected correctness gives the mean BAcc over random deferral draws
    n_def = {d: int((D.dataset == d).sum() // 3) for d in phase4.DATASETS}
    masks = phase4.random_deferral(D, n_def, draws=400, seed=2)
    assert all(masks[:, (D.dataset == d).values].sum(axis=1).tolist() == [n_def[d]] * 400 for d in phase4.DATASETS)
    draws = [phase4.macro_bacc(D, np.where(m, D.pred_F, D.pred_R))[0] for m in masks]
    q = D.dataset.map({d: n_def[d] / (D.dataset == d).sum() for d in phase4.DATASETS}).values
    cR, cF = (D.pred_R == D.label).values, (D.pred_F == D.label).values
    one = {"e": cR + q * (cF - cR.astype(float))}
    ones = np.ones(len(D))
    # with all resample weights equal to 1 the bootstrap formula reduces to the plain BAcc
    exp = np.mean([50 * (one["e"][m & (D.label.values == 1)].mean() + one["e"][m & (D.label.values == 0)].mean())
                   for m in [(D.dataset == d).values for d in phase4.DATASETS]])
    assert abs(np.mean(draws) - exp) < 0.1 and ones.sum() == len(D)


def test_load_refuses_misaligned_labels(env):
    res, _ = env
    f = res / "flan-t5-large" / "dev" / "Wice.csv"
    df = pd.read_csv(f)
    df.loc[0, "label"] = 1 - df.loc[0, "label"]
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        phase4.load("dev")


def test_select_freezes_then_evaluate_reads_it(env):
    _, out = env
    args = type("A", (), {"overwrite": False})()
    phase4.select(args)
    frozen = json.loads((out / "selected_thresholds.json").read_text())
    assert frozen["selected_on"] == "dev" and frozen["primary"]["delta"] <= frozen["secondary"]["delta"] + 1e-12
    curve = pd.read_csv(out / "dev_delta_curve.csv")
    reach = curve[curve.bacc >= frozen["flan_dev_bacc"] - 1e-6]
    if len(reach):  # primary rule: smallest delta reaching Flan-T5 on dev
        assert np.isclose(reach.delta.iloc[0], frozen["primary"]["delta"])
    with pytest.raises(SystemExit):  # cannot silently re-select
        phase4.select(args)
    phase4.evaluate(args)
    s = pd.read_csv(out / "test_summary.csv")
    assert {"RoBERTa alone", "Flan-T5 alone", "Majority vote (3 models)"} <= set(s.approach)
    assert any(a.startswith("Random deferral") for a in s.approach)
    flan = s[s.approach == "Flan-T5 alone"].iloc[0]
    assert np.isclose(flan.speedup_vs_flan, 1.0)
    stats = json.loads((out / "test_statistics.json").read_text())
    assert stats["verdict"]["result"].split(":")[0] in ("positive", "negative")
    with pytest.raises(SystemExit):  # no re-selection once test results exist
        phase4.select(type("A", (), {"overwrite": True})())
    assert (out / "fig_dev_tradeoff.png").exists() and (out / "fig_test_tradeoff.png").exists()


def test_measured_matches_simulation_when_run_equals_it(env):
    res, out = env
    args = type("A", (), {"overwrite": False, "results": res})()
    phase4.select(args)
    frozen = json.loads((out / "selected_thresholds.json").read_text())
    D = phase4.load("test")
    pred, cost, d = phase4.cascade(D, frozen["primary"]["delta"])
    run = res / "cascade" / "test"
    run.mkdir(parents=True)
    M = pd.DataFrame({"idx": D.index, "dataset": D.dataset.values, "label": D.label.values, "prob_r": D.p_R.values,
                      "deferred": d.astype(int), "prob_f": np.where(d, D.p_F.values, np.nan),
                      "prob": np.where(d, D.p_F.values, D.p_R.values), "pred": pred,
                      "latency_r_ms": D.lat_R.values, "latency_f_ms": np.where(d, D.lat_F.values, np.nan),
                      "latency_ms": cost})
    env_rec = {"gpu": "Tesla T4"}
    for name in phase4.DATASETS:
        M[M.dataset == name].to_csv(run / f"{name}.csv", index=False)
        (run / f"{name}.json").write_text(json.dumps({"delta": frozen["primary"]["delta"], "env": env_rec}))
    (res / "cascade" / "runs.jsonl").write_text(json.dumps({"weights_gpu_mb": 4400, "load_seconds": 30}) + "\n")
    phase4.measured(args)
    r = json.loads((out / "measured_vs_simulated.json").read_text())
    assert r["predictions_equal_to_simulation"] == r["n"] == r["deferral_decisions_equal_to_simulation"]
    assert np.isclose(r["bacc_measured"], r["bacc_simulated"])
    assert np.isclose(r["end_to_end_seconds_measured"], r["seconds_simulated"])
