"""Phase 4, end to end: the frozen RoBERTa -> Flan-T5 cascade on one GPU, timed per claim.

Both checkers stay loaded on the GPU. For each claim RoBERTa scores it; if |p_RoBERTa - 0.5| < delta, Flan-T5
scores it too and its prediction is used. delta is read from results/phase4/selected_thresholds.json (chosen on
dev by scripts/phase4.py and frozen before the test evaluation). Data, checkpoints, chunking and the OOM
fallback are the same as in run_checker.py, whose loading functions are reused unchanged.

Output, per dataset:
    <out>/cascade/<split>/<dataset>.csv   idx,dataset,label,prob_r,deferred,prob_f,prob,pred,
                                          n_chunks_r,n_chunks_f,latency_r_ms,latency_f_ms,latency_ms
    <out>/cascade/<split>/<dataset>.json  totals, peak GPU memory, OOM fallbacks per model, versions
    <out>/cascade/runs.jsonl              one line per invocation (load time, GPU memory of both models)
latency_r_ms / latency_f_ms: each stage, measured like Phase 3 (successful attempt only).
latency_ms: the whole claim, end to end (both stages, the decision, and any OOM retries).
A finished dataset is skipped; an unfinished one resumes from <dataset>.part.csv.

Usage:
    python run_cascade.py --split test --out /kaggle/working/minicheck_results
    python run_cascade.py --split test --out smoke --limit 5
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_checker as rc  # noqa: E402  (pinned data/model loading, environment checks)

ROOT = rc.ROOT
CFG = rc.CFG
FIRST, SECOND = "roberta-large", "flan-t5-large"
COLUMNS = ["idx", "dataset", "label", "prob_r", "deferred", "prob_f", "prob", "pred",
           "n_chunks_r", "n_chunks_f", "latency_r_ms", "latency_f_ms", "latency_ms"]
FLUSH_EVERY = 200


def parse_args():
    """Parse the command line: split, output folder, frozen-threshold file, limit, folders, time limit."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--out", required=True, type=Path, help="results root; writes <out>/cascade/")
    p.add_argument("--thresholds", type=Path, default=ROOT / "results" / "phase4" / "selected_thresholds.json")
    p.add_argument("--limit", type=int, help="first N claims per dataset (smoke test)")
    p.add_argument("--models-dir", type=Path, default=Path("/tmp/minicheck_models"))
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--stop-after-min", type=float)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main():
    """Run the frozen cascade end to end: score each claim with RoBERTa, rescore it with Flan-T5 when
    |p - 0.5| < delta, and save per-claim results, stage latencies and peak GPU memory.
    """
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import pandas as pd
    import torch

    frozen = json.loads(args.thresholds.read_text())
    delta = float(frozen["primary"]["delta"])
    frozen_sha = hashlib.sha256(args.thresholds.read_bytes()).hexdigest()[:16]
    env = rc.environment()
    rc.check_environment(env, args.allow_cpu)
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    mb = lambda b: round(b / 2**20, 1)
    t_start = time.time()
    deadline = t_start + 60 * args.stop_after_min if args.stop_after_min else None
    out_root = args.out / "cascade"

    df = rc.load_split(args.split, args.data_dir, args.limit)
    plan = [(name, df[df.dataset == name]) for name in CFG["datasets"]
            if (df.dataset == name).any() and not (out_root / args.split / f"{name}.csv").exists()]
    if not plan:
        print(f"cascade: all datasets of {args.split} already finished", flush=True)
        return
    print(f"cascade (delta {delta}): {len(plan)} datasets, {sum(len(s) for _, s in plan)} claims to score", flush=True)

    # Load both checkers, RoBERTa first; record the GPU memory each one adds
    infs, load_s, weights = {}, {}, {}
    for model in (FIRST, SECOND):
        rc.fetch_model(model, args.models_dir)
        before = torch.cuda.memory_allocated() if cuda else 0
        t0 = time.time()
        scorer = rc.load_scorer(model, args.models_dir.resolve())
        sync()
        load_s[model] = round(time.time() - t0, 1)
        weights[model] = mb(torch.cuda.memory_allocated() - before) if cuda else None
        infs[model] = (scorer, scorer.model)
        scorer.model.chunk_size = CFG["models"][model]["chunk_size"]

    def score_one(model, doc, claim):
        """As run_checker.score_one: on CUDA OOM retry this claim with half the chunk batch size."""
        inf = infs[model][1]
        bs, lost = CFG["batch_size"], 0.0
        try:
            while True:
                inf.batch_size = bs
                sync()
                t = time.perf_counter()
                try:
                    out = inf.inference_per_example(doc, claim)
                    sync()
                    return out["max_support_prob"], len(out["used_chunks"]), 1000 * (time.perf_counter() - t), bs, lost
                except torch.cuda.OutOfMemoryError:
                    pass
                lost += time.perf_counter() - t
                if cuda:
                    torch.cuda.empty_cache()
                if bs == 1:
                    raise RuntimeError(f"{model}: out of GPU memory even with one chunk per batch")
                bs //= 2
        finally:
            inf.batch_size = CFG["batch_size"]

    # Warm-up and check against MiniCheck.score() for both models (3 shortest documents)
    pending = pd.concat([s for _, s in plan])
    first = pending.loc[pending.doc.str.len().nsmallest(3).index]
    for model in (FIRST, SECOND):
        scorer, inf = infs[model]
        _, ref, _, _ = scorer.score(docs=list(first.doc), claims=list(first.claim))
        inf.chunk_size = CFG["models"][model]["chunk_size"]
        ours = [score_one(model, d, c)[0] for d, c in zip(first.doc, first.claim)]
        assert np.allclose(ours, ref, atol=1e-6), f"{model}: per-example {ours} != score() {ref}"
    both_mb = mb(torch.cuda.memory_allocated()) if cuda else None
    print(f"loaded both checkers in {sum(load_s.values()):.1f}s; GPU memory of weights: {weights} "
          f"(total {both_mb} MB); score() checks passed", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "runs.jsonl", "a") as f:
        f.write(json.dumps({"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": args.split,
                            "limit": args.limit, "delta": delta, "thresholds_sha256_16": frozen_sha,
                            "load_seconds": round(sum(load_s.values()), 1), "load_seconds_per_model": load_s,
                            "weights_gpu_mb": both_mb, "weights_gpu_mb_per_model": weights, "env": env}) + "\n")

    for name, sub in plan:
        d = out_root / args.split
        d.mkdir(parents=True, exist_ok=True)
        final, part, part_meta = (d / f"{name}{s}" for s in (".csv", ".part.csv", ".part.json"))
        done = pd.read_csv(part) if part.exists() else pd.DataFrame(columns=COLUMNS)
        prev = json.loads(part_meta.read_text()) if part_meta.exists() else {}
        fb = {FIRST: prev.get(f"batch_fallbacks_{FIRST}", {}), SECOND: prev.get(f"batch_fallbacks_{SECOND}", {})}
        oom_lost = prev.get("oom_retry_seconds", 0.0)
        todo = sub[~sub.idx.isin(set(done.idx))]
        print(f"[{args.split}] {name}: {len(sub)} claims, {len(done)} already done", flush=True)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        rows, t_ds = [], time.time()

        def flush():
            """Append buffered rows to the partial CSV and save fallback and GPU-memory state, so a run can resume."""
            nonlocal rows
            if rows:
                pd.DataFrame(rows, columns=COLUMNS).to_csv(part, mode="a", header=not part.exists(), index=False)
                rows = []
            state = {f"batch_fallbacks_{FIRST}": fb[FIRST], f"batch_fallbacks_{SECOND}": fb[SECOND],
                     "oom_retry_seconds": round(oom_lost, 2)}
            if cuda:
                state["peak_gpu_mb_allocated"] = max(prev.get("peak_gpu_mb_allocated", 0), mb(torch.cuda.max_memory_allocated()))
                state["peak_gpu_mb_reserved"] = max(prev.get("peak_gpu_mb_reserved", 0), mb(torch.cuda.max_memory_reserved()))
            part_meta.write_text(json.dumps(state))

        for k, r in enumerate(todo.itertuples(index=False), 1):
            sync()
            t0 = time.perf_counter()
            p_r, n_r, ms_r, bs_r, lost_r = score_one(FIRST, r.doc, r.claim)
            defer = abs(p_r - 0.5) < delta
            p_f = n_f = ms_f = None
            lost_f = 0.0
            if defer:
                p_f, n_f, ms_f, bs_f, lost_f = score_one(SECOND, r.doc, r.claim)
                if bs_f != CFG["batch_size"]:
                    fb[SECOND][r.idx] = bs_f
            prob = p_f if defer else p_r
            sync()
            total_ms = 1000 * (time.perf_counter() - t0)
            if bs_r != CFG["batch_size"]:
                fb[FIRST][r.idx] = bs_r
            oom_lost += lost_r + lost_f
            rows.append([r.idx, r.dataset, int(r.label), p_r, int(defer), p_f, prob, int(prob > CFG["threshold"]),
                         n_r, n_f, round(ms_r, 3), None if ms_f is None else round(ms_f, 3), round(total_ms, 3)])
            if len(rows) >= FLUSH_EVERY:
                flush()
                print(f"    {len(done) + k}/{len(sub)}  ({(time.time() - t_ds) / k * 1000:.0f} ms/claim)", flush=True)
                if deadline and time.time() > deadline:
                    print("Stopping: --stop-after-min reached. Re-run to resume.", flush=True)
                    return
        flush()

        res = pd.read_csv(part)
        assert set(res.idx) == set(sub.idx) and len(res) == len(sub), f"{name}: incomplete or duplicated rows"
        res = res.set_index("idx").loc[list(sub.idx)].reset_index()
        res.to_csv(final, index=False)
        state = json.loads(part_meta.read_text())
        meta = {"split": args.split, "dataset": name, "n": len(res), "limit": args.limit, "delta": delta,
                "thresholds_sha256_16": frozen_sha, "deferred": int(res.deferred.sum()),
                "end_to_end_seconds": round(res.latency_ms.sum() / 1000, 2),
                "stage_seconds_roberta": round(res.latency_r_ms.sum() / 1000, 2),
                "stage_seconds_flan": round(res.latency_f_ms.fillna(0).sum() / 1000, 2),
                "latency_ms_mean": round(res.latency_ms.mean(), 2), "latency_ms_median": round(res.latency_ms.median(), 2),
                "latency_ms_p95": round(res.latency_ms.quantile(0.95), 2),
                f"n_batch_fallbacks_{FIRST}": len(state[f"batch_fallbacks_{FIRST}"]),
                f"n_batch_fallbacks_{SECOND}": len(state[f"batch_fallbacks_{SECOND}"]),
                **state, "weights_gpu_mb": both_mb,
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": env}
        (d / f"{name}.json").write_text(json.dumps(meta, indent=2))
        part.unlink()
        part_meta.unlink(missing_ok=True)
        print(f"    done: {meta['deferred']} deferred, {meta['end_to_end_seconds']}s end to end, "
              f"peak GPU {state.get('peak_gpu_mb_allocated')} MB", flush=True)
        if deadline and time.time() > deadline:
            print("Stopping: --stop-after-min reached. Re-run to resume.", flush=True)
            return
    print(f"cascade: finished in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
