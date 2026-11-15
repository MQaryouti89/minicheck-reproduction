"""Builds notebooks/minicheck_kaggle.ipynb from the cell list below."""
import json, sys
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r'''
# MiniCheck reproduction on Kaggle (Phases 2–4)

Scores the released **MiniCheck-RoBERTa**, **MiniCheck-DeBERTa** and **MiniCheck-Flan-T5** checkpoints on the
LLM-AggreFact **dev** and **test** splits (the paper's 10 datasets) on one T4 GPU, then compares with Table 2 of
Tang et al. (EMNLP 2024). This is evaluation reproduction with the released checkpoints: no training.
Phase 4 adds one end-to-end run of the RoBERTa → Flan-T5 cascade on the test split, with the threshold δ that
was chosen on dev and frozen in `results/phase4/selected_thresholds.json` (from `scripts/phase4.py select`).

Dataset, model and MiniCheck code versions are pinned in `configs/experiment.json`.

### Before the first run
1. **Settings → Accelerator → GPU T4 x1**. **Settings → Internet → On** (needs a phone-verified Kaggle account).
2. **Settings → Environment → Pin to original environment** (keeps the same Kaggle image for every version).
3. Accept the dataset terms at https://huggingface.co/datasets/lytang/LLM-AggreFact with your Hugging Face account.
4. **Add-ons → Secrets**: add `HF_TOKEN` (a *read* token) and enable it for this notebook. Never paste the token in a cell.
5. Make the code available, either:
   - set `REPO_URL` in cell 1 to the GitHub repo, or
   - upload the repo folder (`scripts/`, `configs/`, `requirements-kaggle.txt`, and for the cascade
     `results/phase4/selected_thresholds.json`) as a Kaggle Dataset and add it as an input.

### How to run: one *Save Version → Save & Run All* per stage
| Version | `STAGES` in cell 1 | Work |
|---|---|---|
| 1 | `["smoke"]` | 5 claims per dataset, both splits, all 3 models + metrics (checks the whole pipeline) |
| 2 | `["roberta-large"]` | 28K claims (15,080 dev + 12,949 test) |
| 3 | `["deberta-v3-large"]` | same |
| 4 | `["flan-t5-large"]` | same (slowest) |
| 5 | `["cascade"]` | Phase 4: frozen RoBERTa → Flan-T5 cascade on the 12,949 test claims, timed end to end (~1 h) |
| any | `[]` | metrics and paper comparison only |

The comparison (cell 12) runs at the end of every version over everything finished so far.

### Recovering from interruptions
- Results are written per **model / split / dataset**, and unfinished datasets are saved every 200 claims.
- Before each new version: **Add Input → Your Work → this notebook → latest version**. Cell 6 copies its
  `minicheck_results/` into `/kaggle/working`, and the runner skips everything already finished.
- Each stage stops itself after `SESSION_HOURS` so the version ends normally and its output is saved
  (Kaggle kills sessions at 12 h). If a stage stopped early, save a new version with the same `STAGES`.
- Use *Save & Run All*, not an interactive session: files in an interactive session are lost when it ends.
''')

md("## 1. Settings (edit before each version)")
code(r'''
STAGES = ["smoke"]        # then ["roberta-large"], ["deberta-v3-large"], ["flan-t5-large"]; [] = metrics only
SPLITS = ["dev", "test"]
REPO_URL = ""             # e.g. "https://github.com/<user>/MiniCheck_Reproduction.git"; "" = use an uploaded Kaggle Dataset
REPO_REF = ""             # optional commit or tag to check out
SESSION_HOURS = 11.0      # stop scoring after this many hours (Kaggle's limit is 12 h)

import glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # one GPU, even if T4 x2 was selected
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"                     # stop transformers importing Kaggle's TensorFlow (slower, noisy load times)
SESSION_START = time.time()
WORK = Path("/kaggle/working")
CODE = Path("/tmp/code")                       # outside /kaggle/working, so old code never shadows new code
RESULTS = WORK / "minicheck_results"           # everything worth keeping goes here
SMOKE = WORK / "smoke"
LOGS = RESULTS / "logs"
MODELS_DIR = Path("/tmp/minicheck_models")     # ~6.3 GB of checkpoints, not saved as output
MODELS = ["roberta-large", "deberta-v3-large", "flan-t5-large"]


def sh(cmd, log=None):
    """Run a command, stream its output here, and optionally append it to a log file."""
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    f = open(log, "a") if log else None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
        if f:
            f.write(line)
    if f:
        f.close()
    if p.wait():
        raise RuntimeError(f"command failed with exit code {p.returncode}")


def status(root=RESULTS):
    cfg = json.loads((CODE / "configs/experiment.json").read_text())
    for m in MODELS:
        done = {s: sum((root / m / s / f"{d}.csv").exists() for d in cfg["datasets"]) for s in SPLITS}
        print(f"  {m:18}" + "  ".join(f"{s}: {n}/10" for s, n in done.items()))
''')

md("## 2. Get the code\nFrom GitHub (`REPO_URL`) or from a Kaggle Dataset attached as input. The dataset may keep the repo's "
   "folders (`scripts/`, `configs/`) or hold the files loose; both work. If nothing is found, the cell lists what is in `/kaggle/input`.")
code(r'''
def find_input(name):
    """All files called `name` under /kaggle/input (follows Kaggle's symlinked mounts)."""
    return sorted(Path(root) / name for root, _, files in os.walk("/kaggle/input", followlinks=True) if name in files)

def show_inputs(max_depth=4):
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        depth = len(Path(root).relative_to("/kaggle/input").parts)
        if depth <= max_depth:
            print("  " * depth + (Path(root).name or "/kaggle/input") + "/", sorted(files)[:8])

shutil.rmtree(CODE, ignore_errors=True)

if REPO_URL:
    sh(["git", "clone", "-q", REPO_URL, CODE])
    if REPO_REF:
        sh(["git", "-C", CODE, "checkout", "-q", REPO_REF])
    version = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    source = f"{REPO_URL} @ {version}"
else:
    runners, configs, reqs = find_input("run_checker.py"), find_input("experiment.json"), find_input("requirements-kaggle.txt")
    roots = sorted({p.parents[1] for p in runners if (p.parents[1] / "configs" / "experiment.json").exists()})
    if len(roots) == 1:                                   # uploaded with the repo's folder structure
        shutil.copytree(roots[0], CODE, ignore=shutil.ignore_patterns("smoke", "ckpts", "*.ipynb"))
        source = str(roots[0])
    elif len(runners) == 1 and len(configs) == 1 and len(reqs) == 1:   # files uploaded loose: rebuild the layout
        (CODE / "scripts").mkdir(parents=True)
        (CODE / "configs").mkdir()
        for p in runners[0].parent.glob("*.py"):
            shutil.copy2(p, CODE / "scripts" / p.name)
        shutil.copy2(configs[0], CODE / "configs" / "experiment.json")
        shutil.copy2(reqs[0], CODE / "requirements-kaggle.txt")
        for p in find_input("selected_thresholds.json")[:1]:     # Phase 4: frozen cascade threshold
            (CODE / "results" / "phase4").mkdir(parents=True)
            shutil.copy2(p, CODE / "results" / "phase4" / p.name)
        for p in find_input("roberta-large_preds.csv")[:1]:      # optional: laptop run, for the hardware check
            (CODE / "results" / "roberta-large").mkdir(parents=True)
            shutil.copy2(p, CODE / "results" / "roberta-large" / p.name)
        source = f"loose files in {runners[0].parent}"
    else:
        show_inputs()
        raise RuntimeError(
            f"Could not find exactly one copy of the code under /kaggle/input "
            f"(run_checker.py: {runners}, experiment.json: {configs}, requirements-kaggle.txt: {reqs}).\n"
            "Fix: right panel > Add Input > Your Work > Datasets > select the dataset with the code, "
            "or set REPO_URL in cell 1. The dataset must contain scripts/run_checker.py, scripts/metrics.py, "
            "configs/experiment.json and requirements-kaggle.txt.")

# Record exactly which code produced the outputs
RESULTS.mkdir(parents=True, exist_ok=True)
hashes = {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
          for p in sorted(CODE.glob("scripts/*.py")) + sorted(CODE.glob("configs/*.json")) + [CODE / "requirements-kaggle.txt"]
          + sorted(CODE.glob("results/phase4/selected_thresholds.json"))}
(RESULTS / "code_version.json").write_text(json.dumps({"source": source, "sha256_16": hashes}, indent=2))
print(source)
print(json.dumps(hashes, indent=2))
''')

md("## 3. Install pinned dependencies\nKaggle's PyTorch is kept as is. MiniCheck is installed with `--no-deps` so it cannot upgrade torch or transformers.")
code(r'''
cfg = json.loads((CODE / "configs/experiment.json").read_text())
sh([sys.executable, "-m", "pip", "install", "-q", "-r", CODE / "requirements-kaggle.txt"])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    f"minicheck @ git+https://github.com/Liyan06/MiniCheck.git@{cfg['minicheck_commit']}"])
''')

md("## 4. Check the environment\nFails if no GPU is visible. Versions are saved with every result file as well.")
code(r'''
sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
sh([sys.executable, "-c", """
import platform, torch, transformers, tokenizers, huggingface_hub, accelerate, nltk, pandas, minicheck
print('python', platform.python_version(), '| torch', torch.__version__, 'cuda', torch.version.cuda)
print('transformers', transformers.__version__, '| tokenizers', tokenizers.__version__,
      '| huggingface_hub', huggingface_hub.__version__, '| accelerate', accelerate.__version__,
      '| nltk', nltk.__version__, '| pandas', pandas.__version__)
assert torch.cuda.is_available(), 'No GPU: set Accelerator = GPU T4 x1'
name = torch.cuda.get_device_name(0)
print('GPU:', name, '| visible GPUs:', torch.cuda.device_count())
if 'T4' not in name:
    print('WARNING: not a T4. Keep every model on the same GPU type, or runtimes are not comparable.')
"""])
with open(RESULTS / f"pip_freeze_{time.strftime('%Y%m%d-%H%M%S')}.txt", "w") as f:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
''')

md("## 5. Hugging Face access\nThe dataset is gated; the token is read from Kaggle Secrets and passed to the scripts as `HF_TOKEN`.")
code(r'''
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("HF_TOKEN set")
''')

md("## 6. Restore results from the previous version\nAttach only the **latest** previous version of this notebook as input. Existing files are never overwritten.")
code(r'''
copied = 0
for prev in glob.glob("/kaggle/input/**/minicheck_results", recursive=True):
    prev = Path(prev)
    for f in prev.rglob("*"):
        dst = RESULTS / f.relative_to(prev)
        if f.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            copied += 1
print(f"restored {copied} files")
status()
''')

md("## 7. Smoke test\n5 claims per dataset and split for each model. Downloads all three checkpoints and runs the metrics on the result. Writes to `smoke/`, separate from the real results.")
code(r'''
if "smoke" in STAGES:
    shutil.rmtree(SMOKE, ignore_errors=True)
    for m in MODELS:
        sh([sys.executable, CODE / "scripts/run_checker.py", m, "--splits", *SPLITS, "--out", SMOKE,
            "--models-dir", MODELS_DIR, "--limit", 5])
    sh([sys.executable, CODE / "scripts/metrics.py", SMOKE])
    status(SMOKE)
''')


def model_cell(n, model, title):
    """Add the notebook cells for one checker stage: a heading and a cell that runs run_checker.py."""
    md(f"## {n}. {title}")
    code(f'''
if "{model}" in STAGES:
    LOGS.mkdir(parents=True, exist_ok=True)
    minutes_left = SESSION_HOURS * 60 - (time.time() - SESSION_START) / 60
    sh([sys.executable, CODE / "scripts/run_checker.py", "{model}", "--splits", *SPLITS, "--out", RESULTS,
        "--models-dir", MODELS_DIR, "--stop-after-min", round(minutes_left)], log=LOGS / "{model}.log")
status()
''')


model_cell(8, "roberta-large", "MiniCheck-RoBERTa-Large")
model_cell(9, "deberta-v3-large", "MiniCheck-DeBERTa-v3-Large")
model_cell(10, "flan-t5-large", "MiniCheck-Flan-T5-Large")

md("## 11. Phase 4: end-to-end cascade (test split)\n"
   "Both checkers stay loaded on the T4. RoBERTa scores every claim; claims with |p − 0.5| < δ also go to Flan-T5. "
   "δ comes only from the frozen file, whose hash is checked first. Writes `minicheck_results/cascade/test/`. "
   "Compare with the simulation afterwards, locally: `python scripts/phase4.py measured --results <downloaded minicheck_results>`.")
code(r'''
FROZEN_SHA256_16 = "ce2102326bc8c686"   # results/phase4/selected_thresholds.json, frozen on dev 2026-11-15
if "cascade" in STAGES:
    thr = CODE / "results/phase4/selected_thresholds.json"
    assert thr.exists(), "results/phase4/selected_thresholds.json is missing from the uploaded code"
    got = hashlib.sha256(thr.read_bytes()).hexdigest()[:16]
    assert got == FROZEN_SHA256_16, f"threshold file changed: {got} != {FROZEN_SHA256_16}"
    print("frozen delta:", json.loads(thr.read_text())["primary"]["delta"])
    LOGS.mkdir(parents=True, exist_ok=True)
    minutes_left = SESSION_HOURS * 60 - (time.time() - SESSION_START) / 60
    sh([sys.executable, CODE / "scripts/run_cascade.py", "--split", "test", "--out", RESULTS, "--thresholds", thr,
        "--models-dir", MODELS_DIR, "--stop-after-min", round(minutes_left)], log=LOGS / "cascade.log")
cdir = RESULTS / "cascade/test"
metas = [json.loads(p.read_text()) for p in sorted(cdir.glob("*.json"))] if cdir.exists() else []
if metas:
    n = sum(m["n"] for m in metas); dfr = sum(m["deferred"] for m in metas)
    print(f"cascade: {len(metas)}/10 datasets, {n} claims, {dfr} deferred ({100 * dfr / n:.1f}%), "
          f"{sum(m['end_to_end_seconds'] for m in metas):.0f} s end to end, "
          f"peak GPU {max(m.get('peak_gpu_mb_allocated', 0) for m in metas):.0f} MB allocated, "
          f"Flan-T5 batch fallbacks {sum(m['n_batch_fallbacks_flan-t5-large'] for m in metas)}")
''')

md("## 12. Metrics and comparison with the paper\nCPU only. Recomputes everything from the saved predictions. "
   "If the repo contains the earlier laptop RoBERTa predictions, it also reports how many predictions changed between the laptop GPU and the T4.")
code(r'''
legacy = CODE / "results/roberta-large/roberta-large_preds.csv"
if any(RESULTS.glob("*/*/*.csv")):
    sh([sys.executable, CODE / "scripts/metrics.py", RESULTS] + (["--legacy", legacy] if legacy.exists() else []))
    from IPython.display import Markdown, display
    display(Markdown((RESULTS / "metrics/comparison_test.md").read_text()))
else:
    print("no finished results yet")
''')

md("## 13. Package the outputs\n`minicheck_results/` is saved as this version's output. The zip is a convenience copy for downloading.")
code(r'''
shutil.make_archive(str(WORK / "minicheck_results"), "zip", WORK, "minicheck_results")
print(f"{WORK / 'minicheck_results.zip'}: {(WORK / 'minicheck_results.zip').stat().st_size / 1e6:.1f} MB")
status()
''')

for c in cells:  # notebook format stores source as a list of lines
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", sys.argv[1], len(cells), "cells")
