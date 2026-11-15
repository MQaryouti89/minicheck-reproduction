"""Score LLM-AggreFact (dev and/or test) with one released MiniCheck checker, resumably.

Uses MiniCheck's own per-example inference (Inferencer.inference_per_example, the function that
MiniCheck.score() loops over), so predictions are identical to score(), but each example can be timed.
Dataset, model and MiniCheck code are pinned in configs/experiment.json.

Output, per model / split / dataset:
    <out>/<model>/<split>/<dataset>.csv    idx,dataset,label,prob,pred,n_chunks,latency_ms
    <out>/<model>/<split>/<dataset>.json   runtime, GPU memory, versions for that dataset
    <out>/<model>/runs.jsonl               one line per invocation (model load time, weights memory)
A finished dataset is skipped. An unfinished one resumes from <dataset>.part.csv.

Usage:
    python run_checker.py roberta-large --splits dev test --out results/kaggle-t4
    python run_checker.py roberta-large --out smoke --limit 5        # 5 claims per dataset
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "configs" / "experiment.json").read_text())
COLUMNS = ["idx", "dataset", "label", "prob", "pred", "n_chunks", "latency_ms"]
FLUSH_EVERY = 200  # rows appended to <dataset>.part.csv at a time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=list(CFG["models"]))
    p.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"])
    p.add_argument("--out", required=True, type=Path, help="results root folder")
    p.add_argument("--limit", type=int, help="score only the first N claims of each dataset (smoke test)")
    p.add_argument("--models-dir", type=Path, default=Path("/tmp/minicheck_models"),
                   help="where pinned checkpoints are downloaded (keep outside Kaggle's /kaggle/working)")
    p.add_argument("--data-dir", type=Path, help="read <split>.parquet from here instead of Hugging Face")
    p.add_argument("--stop-after-min", type=float,
                   help="stop cleanly (progress saved) after this many minutes, e.g. before Kaggle's 12 h limit")
    p.add_argument("--allow-cpu", action="store_true", help="allow running without a GPU (testing only)")
    return p.parse_args()


# ---------------------------------------------------------------- data and model
def load_split(split, data_dir, limit):
    import pandas as pd
    if data_dir:
        path = Path(data_dir) / f"{split}.parquet"
    else:
        from huggingface_hub import hf_hub_download
        ds = CFG["dataset"]
        path = hf_hub_download(ds["repo"], ds["files"][split], repo_type="dataset", revision=ds["revision"])
    df = pd.read_parquet(path)
    # idx = row position in the pinned parquet file, so it is the same for every model and run
    df.insert(0, "idx", [f"{split}-{i:05d}" for i in range(len(df))])
    df = df[df.dataset.isin(CFG["datasets"])]
    counts = df.dataset.value_counts().to_dict()
    expected = CFG["expected_counts"][split]
    if counts != expected:
        msg = f"{split}: claim counts differ from the paper (Table 11).\n  got      {counts}\n  expected {expected}"
        if limit is None and data_dir is None:
            sys.exit(msg)
        print("WARNING " + msg, flush=True)
    if limit:
        df = df.groupby("dataset").head(limit)
    return df.reset_index(drop=True)


def fetch_model(model, models_dir):
    """Download the pinned checkpoint to <models_dir>/<repo>, e.g. /tmp/minicheck_models/lytang/MiniCheck-RoBERTa-Large."""
    from huggingface_hub import snapshot_download
    m = CFG["models"][model]
    local = Path(models_dir) / m["repo"]
    snapshot_download(m["repo"], revision=m["revision"], local_dir=str(local), allow_patterns=CFG["model_files"])
    return local


def load_scorer(model, models_dir):
    """Build MiniCheck so that it loads the pinned local copy.

    MiniCheck hard-codes the repo id (e.g. 'lytang/MiniCheck-RoBERTa-Large'). transformers treats that
    string as a local path when a folder with that relative path exists, so we chdir into models_dir.
    """
    import nltk
    for res in ("punkt", "punkt_tab"):  # MiniCheck only fetches 'punkt'; nltk >= 3.8.2 sent_tokenize needs 'punkt_tab'
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    from minicheck.minicheck import MiniCheck

    repo = CFG["models"][model]["repo"]
    cwd = os.getcwd()
    os.chdir(models_dir)
    try:
        assert Path(repo, "config.json").is_file(), f"pinned checkpoint missing in {models_dir / repo}"
        scorer = MiniCheck(model_name=model, batch_size=CFG["batch_size"], cache_dir=None)
    finally:
        os.chdir(cwd)
    assert scorer.model.model.name_or_path == repo, scorer.model.model.name_or_path
    return scorer


# ---------------------------------------------------------------- environment
def environment():
    from importlib import metadata
    import torch
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    try:
        direct = json.loads(metadata.distribution("minicheck").read_text("direct_url.json") or "{}")
        mc_commit = direct.get("vcs_info", {}).get("commit_id")
    except Exception:
        mc_commit = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu,
        "gpu_count": torch.cuda.device_count(),
        **{p: ver(p) for p in ["transformers", "tokenizers", "huggingface-hub", "accelerate", "nltk", "pandas"]},
        "minicheck_commit": mc_commit,
    }


def check_environment(env, allow_cpu):
    if env["minicheck_commit"] not in (None, CFG["minicheck_commit"]):
        sys.exit(f"minicheck is installed from commit {env['minicheck_commit']}, expected {CFG['minicheck_commit']}")
    if env["minicheck_commit"] is None:
        print("WARNING could not read the installed minicheck commit", flush=True)
    if env["gpu"] == "cpu" and not allow_cpu:
        sys.exit("No GPU visible. On Kaggle choose Accelerator = GPU T4 x1 (or pass --allow-cpu for testing).")
    if env["gpu_count"] > 1:
        sys.exit("More than one GPU visible; MiniCheck would split the model across them. "
                 "Set CUDA_VISIBLE_DEVICES=0 or choose GPU T4 x1.")


# ---------------------------------------------------------------- scoring
def main():
    args = parse_args()
    # Less fragmentation of GPU memory; affects only allocation, not results
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import pandas as pd
    import torch

    env = environment()
    check_environment(env, args.allow_cpu)
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    mb = lambda b: round(b / 2**20, 1)
    t_start = time.time()
    deadline = t_start + 60 * args.stop_after_min if args.stop_after_min else None
    threshold = CFG["threshold"]
    mcfg = CFG["models"][args.model]
    model_dir = args.out / args.model

    # Work out what is left before loading anything heavy
    plan = []
    for split in args.splits:
        df = load_split(split, args.data_dir, args.limit)
        for name in CFG["datasets"]:
            if (df.dataset == name).any() and not (model_dir / split / f"{name}.csv").exists():
                plan.append((split, name, df[df.dataset == name]))
    if not plan:
        print(f"{args.model}: all datasets already finished for {args.splits}", flush=True)
        return
    print(f"{args.model}: {len(plan)} split/dataset parts to score, "
          f"{sum(len(s) for *_, s in plan)} claims (before resuming partial files)", flush=True)

    t0 = time.time()
    local = fetch_model(args.model, args.models_dir)
    t_download = time.time() - t0
    t0 = time.time()
    scorer = load_scorer(args.model, args.models_dir.resolve())
    sync()
    t_load = time.time() - t0
    weights_mb = mb(torch.cuda.memory_allocated()) if cuda else None
    inf = scorer.model  # minicheck.inference.Inferencer
    inf.chunk_size = mcfg["chunk_size"]  # normally set inside MiniCheck.score(); we call the per-example function directly

    def score_one(doc, claim):
        """Score one claim. Returns prob, n_chunks, latency_ms, chunk batch size used, seconds lost to OOM.

        MiniCheck scores a document's chunks in batches of `batch_size` (16). A chunk can be up to
        max_model_len (2048) tokens when one 'sentence' is very long, and 16 such chunks do not fit on a
        16 GB T4 for DeBERTa/Flan-T5 (the authors used a 48 GB A6000). On out-of-memory we retry that claim
        with half the batch size. Each chunk is still scored on its own (only the grouping changes), and
        the batch size used is recorded. latency_ms is the successful attempt only.
        """
        bs, lost = CFG["batch_size"], 0.0
        try:
            while True:
                inf.batch_size = bs
                sync()
                t = time.perf_counter()
                try:
                    out = inf.inference_per_example(doc, claim)
                    sync()
                    ms = 1000 * (time.perf_counter() - t)
                    return out["max_support_prob"], len(out["used_chunks"]), ms, bs, lost
                except torch.cuda.OutOfMemoryError:
                    pass  # leave the except block first so the failed batch's tensors can be freed
                lost += time.perf_counter() - t
                if cuda:
                    torch.cuda.empty_cache()
                if bs == 1:
                    raise RuntimeError("out of GPU memory even with one chunk per batch")
                bs //= 2
        finally:
            inf.batch_size = CFG["batch_size"]

    # Warm-up (CUDA kernels, caches) and a check that our loop reproduces MiniCheck.score() exactly
    # Use the 3 shortest documents still to score, so this check cannot itself run out of GPU memory
    pending = pd.concat([s for *_, s in plan])
    first = pending.loc[pending.doc.str.len().nsmallest(3).index]
    _, ref_probs, _, _ = scorer.score(docs=list(first.doc), claims=list(first.claim))
    inf.chunk_size = mcfg["chunk_size"]
    ours = [score_one(d, c)[0] for d, c in zip(first.doc, first.claim)]
    assert np.allclose(ours, ref_probs, atol=1e-6), f"per-example probs {ours} != score() probs {ref_probs}"
    print(f"loaded {mcfg['repo']}@{mcfg['revision'][:7]} from {local} in {t_load:.1f}s "
          f"(download/check {t_download:.1f}s), weights on GPU: {weights_mb} MB; score() check passed", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps({"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "splits": args.splits, "limit": args.limit, "load_seconds": round(t_load, 1),
                            "download_seconds": round(t_download, 1), "weights_gpu_mb": weights_mb,
                            "model_revision": mcfg["revision"], "dataset_revision": CFG["dataset"]["revision"],
                            "env": env}) + "\n")

    for split, name, sub in plan:
        out_dir = model_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        final, part, part_meta = (out_dir / f"{name}{s}" for s in (".csv", ".part.csv", ".part.json"))
        done = pd.read_csv(part) if part.exists() else pd.DataFrame(columns=COLUMNS)
        prev = json.loads(part_meta.read_text()) if part_meta.exists() else {}
        fallbacks = prev.get("batch_fallbacks", {})  # idx -> smaller chunk batch size used after OOM
        oom_lost = prev.get("oom_retry_seconds", 0.0)
        todo = sub[~sub.idx.isin(set(done.idx))]
        print(f"[{split}] {name}: {len(sub)} claims, {len(done)} already done", flush=True)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        rows, t_ds = [], time.time()

        def flush():
            nonlocal rows
            if rows:
                pd.DataFrame(rows, columns=COLUMNS).to_csv(part, mode="a", header=not part.exists(), index=False)
                rows = []
            state = {"batch_fallbacks": fallbacks, "oom_retry_seconds": round(oom_lost, 2)}
            if cuda:
                state["peak_gpu_mb_allocated"] = max(prev.get("peak_gpu_mb_allocated", 0), mb(torch.cuda.max_memory_allocated()))
                state["peak_gpu_mb_reserved"] = max(prev.get("peak_gpu_mb_reserved", 0), mb(torch.cuda.max_memory_reserved()))
            part_meta.write_text(json.dumps(state))

        for k, r in enumerate(todo.itertuples(index=False), 1):
            prob, n_chunks, ms, bs, lost = score_one(r.doc, r.claim)
            if bs != CFG["batch_size"]:
                fallbacks[r.idx] = bs
                oom_lost += lost
                print(f"    {r.idx}: out of GPU memory with {CFG['batch_size']} chunks per batch "
                      f"({n_chunks} chunks); scored with {bs}", flush=True)
            rows.append([r.idx, r.dataset, int(r.label), prob, int(prob > threshold), n_chunks, round(ms, 3)])
            if len(rows) >= FLUSH_EVERY:
                flush()
                print(f"    {len(done) + k}/{len(sub)}  ({(time.time() - t_ds) / k * 1000:.0f} ms/claim)", flush=True)
                if deadline and time.time() > deadline:
                    print(f"Stopping: --stop-after-min {args.stop_after_min} reached. Re-run to resume.", flush=True)
                    return
        flush()

        res = pd.read_csv(part)
        assert set(res.idx) == set(sub.idx) and len(res) == len(sub), f"{name}: incomplete or duplicated rows"
        res = res.set_index("idx").loc[list(sub.idx)].reset_index()  # original order
        res.to_csv(final, index=False)
        lat = res.latency_ms
        state = json.loads(part_meta.read_text()) if part_meta.exists() else {}
        peak = {k: v for k, v in state.items() if k.startswith("peak_")}
        meta = {"model": args.model, "split": split, "dataset": name, "n": len(res), "limit": args.limit,
                "n_batch_fallbacks": len(state.get("batch_fallbacks", {})),
                "batch_fallbacks": state.get("batch_fallbacks", {}),
                "oom_retry_seconds": state.get("oom_retry_seconds", 0.0),
                "inference_seconds": round(lat.sum() / 1000, 2),
                "latency_ms_mean": round(lat.mean(), 2), "latency_ms_median": round(lat.median(), 2),
                "latency_ms_p95": round(lat.quantile(0.95), 2), "n_chunks_mean": round(res.n_chunks.mean(), 3),
                **peak, "weights_gpu_mb": weights_mb, "chunk_size": mcfg["chunk_size"],
                "max_model_len": inf.max_model_len, "batch_size": CFG["batch_size"], "threshold": threshold,
                "model_revision": mcfg["revision"], "dataset_revision": CFG["dataset"]["revision"],
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": env}
        (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
        part.unlink()
        part_meta.unlink(missing_ok=True)
        print(f"    done: {meta['inference_seconds']}s, {meta['latency_ms_mean']} ms/claim, "
              f"peak GPU {peak.get('peak_gpu_mb_allocated')} MB", flush=True)
        if deadline and time.time() > deadline:
            print(f"Stopping: --stop-after-min {args.stop_after_min} reached. Re-run to resume.", flush=True)
            return
    print(f"{args.model}: finished in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
