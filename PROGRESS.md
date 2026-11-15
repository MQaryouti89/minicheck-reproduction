# Project progress

Last updated: 2026-11-15

| Phase | Status |
|---|---|
| 1. Review and verification | **Done** |
| 2. Kaggle preparation | **Done** |
| 3. Reproduce the three models | **Done.** All three reproduce the paper on one Kaggle T4 (test averages within 0.03) |
| 4. Adaptive cascade | **Done, reviewed and approved.** Pre-registered verdict: negative (−0.21 [−1.22, +0.82] vs Flan-T5, measured 1.47× faster) |
| 5. Error analysis, reproducibility, final preparation | **Technical work done; report drafted.** Pending (student): GitHub push, review of the report, presentation, submission |

## 25-step tracker

| # | Phase | Step | Status |
|---|---|---|---|
| 1 | 1 | Review the paper (ACL, arXiv v1/v2) and the official repo at `b58b9fa` | Done |
| 2 | 1 | Recompute the laptop RoBERTa result (73.14) | Done |
| 3 | 1 | Confirm the official aggregation (per-dataset BAcc, unweighted mean, threshold 0.5) | Done |
| 4 | 1 | Investigate the 72.7 vs 73.12 RoBERTa average | Done (paper inconsistency documented) |
| 5 | 1 | Confirm the splits against Table 11 (dev 15,080, test 12,949) | Done |
| 6 | 2 | Pin code, dataset, model and package versions | Done |
| 7 | 2 | Runner and metrics scripts (per-claim `idx`, latency, resume, GPU memory) | Done |
| 8 | 2 | One Kaggle notebook | Done |
| 9 | 2 | Local CPU tests | Done |
| 10 | 2 | Kaggle smoke test, all three models | Done |
| 11 | 3 | RoBERTa, dev + test (version 1) | Done |
| 12 | 3 | Diagnose and fix the T4 out-of-memory failure (version 2) | Done |
| 13 | 3 | DeBERTa and Flan-T5, dev + test (version 4) | Done |
| 14 | 3 | Verify metrics locally; laptop vs T4 check (0 flips) | Done |
| 15 | 3 | Final three-model comparison with Table 2 | Done |
| 16 | 4 | Input checks and pre-registered plan (`configs/phase4.json`) | Done |
| 17 | 4 | Dev analysis; δ frozen (0.19 / 0.41) | Done |
| 18 | 4 | One test evaluation: all comparisons, metrics, bootstrap CIs | Done |
| 19 | 4 | Cascade error analysis | Done |
| 20 | 4 | End-to-end Kaggle run (version 6) and measured vs simulated | Done |
| 21 | 5 | Final error analysis (FP/FN, agreement, confident errors, datasets, length/cost, claim types, examples) | Done |
| 22 | 5 | Final tables, figures and consistency check (32/32 PASS) | Done |
| 23 | 5 | Repository review: cleanup, `.gitignore`, secrets/text scan, 19 tests, README; local git repository and commits | Done. Push to the private GitHub repository `MQaryouti89/minicheck-reproduction` pending (student) |
| 24 | 5 | Literature: 16 verified references, relevance notes, five suggested papers | Done |
| 25 | 5 | Submission checklist, AI-use statement, research report | Checklist done; report drafted (`../Report/AI504_Assignment2_MiniCheck_Report.docx`, not in the repository), awaiting the student's review; presentation not started |

## Phase 1: Review and verification

- [x] Paper (ACL Anthology version, arXiv v1 and v2) and official repo at commit `b58b9fa` reviewed
- [x] Laptop RoBERTa result recomputed from `roberta-large_preds.csv`: 73.14 average; 9 of 10 datasets within rounding. ClaimVerify is +0.18, about one flipped unsupported claim (one flip = 0.167)
- [x] Official aggregation: BAcc per dataset, unweighted mean, threshold 0.5 (authors' `benchmark_evaluation_demo.ipynb`)
- [x] 72.7 discrepancy investigated. Unweighted mean matches 20 of 22 rows in Tables 2, 5, 7, 8 and 9 to within rounding. Both exceptions are RoBERTa (Table 2: −0.42; Table 7: −0.28). Size-weighted mean (72.18) does not explain it either
- [x] Splits confirmed against paper Table 11: dev 15,080 and test 12,949 claims over 10 datasets
- [x] Bug fixed: `--limit` in `minicheck_eval.py` crashed on pandas 3.x
- [ ] Optional: email the authors about the RoBERTa average (Tables 2 and 7)

## Phase 2: Kaggle preparation

- [x] Pinned versions: MiniCheck code, dataset revision, three model revisions (`configs/experiment.json`)
- [x] Dependencies: `requirements-kaggle.txt` on top of Kaggle's torch; MiniCheck installed with `--no-deps`
- [x] `scripts/run_checker.py`: dev/test, stable `idx`, per-example `n_chunks` and `latency_ms`, per-model/split/dataset files, resume, time limit, GPU memory, environment record
- [x] `scripts/metrics.py`: BAcc, P/R/F1 (both classes), confusion matrix, runtime, GPU memory, paper comparison, laptop-vs-T4 check
- [x] Notebook rewritten: one sequence of 12 cells
- [x] Local CPU test with the real pinned RoBERTa checkpoint on synthetic LLM-AggreFact-shaped data (torch 2.15 CPU, transformers 4.57.6):
  - the per-example loop matches `score()`, and long documents are chunked (7 chunks)
  - `idx` survives filtering out RAGTruth
  - resume after an interrupted dataset gives identical results
  - stopping at the time limit saves progress
  - metrics, paper comparison and laptop check run
- [ ] Not testable locally: real gated data, GPU memory figures, Kaggle input/output paths (checked by the smoke version)
- [x] Code uploaded as Kaggle Dataset `minicheck-code`; `HF_TOKEN` secret added
- [x] Smoke test on Kaggle Tesla T4 (2026-11-14), all three models:
  - pinned data downloaded, and dev/test claim counts match Table 11 (no warning)
  - each model loaded from its pinned revision, and the `score()` check passed
  - load time / GPU memory of the weights / peak GPU memory, by model:
    - RoBERTa: 32.5 s / 1,356 MiB / ~1.5 thousand MiB
    - DeBERTa: 14.3 s / 1,660 MiB / ~2.7 thousand MiB
    - Flan-T5: 18.4 s / 2,990 MiB / ~3.9 thousand MiB
    - (smoke-test peaks were written as MiB ÷ 1,000 "GB"; see the unit correction in Phase 5)
  - estimated full dev + test time on T4 (from the smoke timings, rough): RoBERTa ~1.2 h, DeBERTa ~1.8 h, Flan-T5 ~2.4 h
- [x] After the smoke test: silenced sklearn warnings on one-class smoke samples (metrics unchanged); `USE_TF=0` so TensorFlow is not imported during the timed model load

## Phase 3: Reproduce the three models (Kaggle T4)

| Model | dev | test | Avg BAcc (test) | Paper row mean | Datasets matching | Runtime (dev + test) | ms/claim | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|
| RoBERTa-Large | ✓ 71.78 | ✓ 73.14 | **73.14** | 73.12 | 9/10 (ClaimVerify +0.18) | 88 min (2,926 s + 2,361 s) | 194 / 182 | 1,733 |
| DeBERTa-v3-Large | ✓ 70.99 | ✓ 72.56 | **72.56** | 72.53 | 9/10 (AggreFact-CNN +0.12) | 140 min (4,614 s + 3,762 s) | 306 / 290 | 14,704 |
| Flan-T5-Large | ✓ 72.38 | ✓ 74.71 | **74.71** | 74.70 | 8/10 (Wice +0.22, ExpertQA −0.13) | 184 min (6,064 s + 4,954 s) | 402 / 383 | 14,356 |

- [x] Kaggle version 1: RoBERTa, dev + test (2026-11-14, 1 h 29 min session, 88.6 min scoring)
  - Environment: Python 3.13.16, torch 2.11.0+cu130, transformers 4.57.6, tokenizers 0.22.2, nltk 3.10.4. Code hashes match the uploaded code.
  - Test: 73.14 macro BAcc, the same as the laptop run, the paper row mean (73.12) and Table 2 for 9 of 10 datasets.
  - **Laptop (RTX 5070 Ti) vs T4: 0 of 12,949 predictions differ; largest probability difference 5.4e-6.** So the ClaimVerify +0.18 is not caused by the GPU. It comes from something both of our runs share and the authors' run did not (library versions or their environment). That is 1 changed prediction near the threshold: 7 ClaimVerify probabilities are within 0.01 of 0.5, and 4 of them are unsupported claims.
  - The T4 is 2.3× slower than the laptop GPU on test (2,361 s vs 1,041 s). Time per claim follows the number of chunks: ~39 ms with 1 chunk (Reveal, FactCheck-GPT), ~500 ms with 5–6 (Wice, ClaimVerify).
  - Local re-run of `metrics.py` on the downloaded files gives the same numbers as Kaggle's.
- [x] Kaggle version 2: DeBERTa + Flan-T5. **Failed: CUDA out of memory** on DeBERTa dev Wice after 4 of 10 dev datasets and 200 Wice claims.
  - Cause: one sentence-split chunk of up to 2,048 tokens × 16 chunks per batch; DeBERTa's attention needed 4 GiB more than the T4's 15 GiB. The authors used a 48 GB A6000.
  - Fix (`run_checker.py`): on OOM, retry only that claim with 8/4/2/1 chunks per batch, and record it (`batch_fallbacks`).
  - Tested with tiny random-weight models of all three types (local RAM was too low for the real checkpoints): batch size changes the probability by 0 (RoBERTa, DeBERTa) and at most 1.4e-6 (Flan-T5). The simulated-OOM retry survives an interruption and resume.
  - Also fixed: the start-up `score()` check now uses the 3 shortest documents, so it cannot run out of memory itself.
- [x] Kaggle version 3: cancelled after about 3 h in the queue (never started)
- [x] Kaggle version 4: DeBERTa + Flan-T5, dev + test, with the OOM fix (2026-11-14, 5 h 25 min session). Version 1's output was attached, so all three models are in one output.
  - Code hashes match the upload (`run_checker.py` `412f0d4c59ed1b95`). Same environment as version 1 for every model and dataset: Python 3.13.16, torch 2.11.0+cu130, transformers 4.57.6, tokenizers 0.22.2, MiniCheck `b58b9fa`, Tesla T4.
  - All 6 model/split runs complete: 10/10 datasets, counts match Table 11, `idx` unique.
  - OOM fallback used for 5 of 84,087 scored claims, each once, at 8 chunks per batch: DeBERTa dev-04961 (Wice), test-07944 (ExpertQA); Flan-T5 test-05014, test-05016, test-05699 (ClaimVerify). None is in a dataset that differs from the paper.
  - Peak GPU memory reached 14,704 MiB = 14.4 GiB (DeBERTa) of the T4's 15 GiB, so the fallback was needed.
- [x] Local re-run of `metrics.py` gives the same numbers as Kaggle's
- [x] Final comparison with Table 2 for all three models (`metrics/comparison_test.md`):
  - test average vs paper row mean: RoBERTa +0.02, DeBERTa +0.03, Flan-T5 +0.01
  - 26 of 30 dataset scores match after rounding. Each of the other 4 is explained by 1 changed prediction (2 for Flan-T5 ExpertQA)
  - ranking is the same as the paper: Flan-T5 > RoBERTa > DeBERTa
  - cost on the T4 (test): Flan-T5 383 ms/claim, 2.1× RoBERTa's 182 ms, for +1.6 BAcc. This is the trade-off Phase 4 targets
- [x] Laptop vs T4 RoBERTa check (`metrics/hardware_check.json`)
- [x] Kaggle version 1 output copied to `results/kaggle-t4/minicheck_results/`

## Phase 4: Adaptive cascade (done; reviewed and approved 2026-11-15)

Question: can RoBERTa first, with only uncertain claims sent to Flan-T5, (1) beat Flan-T5's test BAcc (74.71), or (2) match it with clearly less inference time? A negative result is reported as such.

**Step 1: check the inputs.** Done 2026-11-15, all checks pass:
- [x] All 6 model/split prediction sets have `idx, dataset, label, prob, pred, n_chunks, latency_ms`.
  - No missing values.
  - `pred == (prob > 0.5)`, probabilities are within [0, 1], and every latency is > 0.
- [x] The same `idx` set, labels and datasets for all three models (dev 15,080, test 12,949).
- [x] `latency_ms` is the CUDA-synchronised wall time of MiniCheck's `inference_per_example` for one claim: sentence split, chunking, tokenisation and forward pass. It excludes data loading and model loading.
- [x] Headroom on **dev**: RoBERTa is wrong on 29.2% of claims. On 10.4%, RoBERTa is wrong and Flan-T5 is right; on 7.1%, RoBERTa is right and Flan-T5 is wrong.
- [x] Per-claim latencies of RoBERTa and Flan-T5 correlate (r = 0.95). Both grow with document length, so deferring long documents is expensive.
- Note: on dev, Flan-T5 beats RoBERTa by only 0.60 BAcc (72.38 vs 71.78). On test the gap is 1.57. A threshold matched to Flan-T5 on dev may therefore under-defer on test.

**Step 2: fix the plan before any results** (`configs/phase4.json`, 2026-11-15; the four choices were approved by the user before the dev analysis):
- [x] Cascade: defer if |p_RoBERTa − 0.5| < δ.
  - δ is a single global value, on the grid 0 to 0.5 in steps of 0.005.
  - Each model's decision uses threshold 0.5.
- [x] Primary operating point: the smallest δ whose **dev** macro BAcc is at least Flan-T5's dev BAcc.
  - If no δ reaches it: the δ with the best dev BAcc, reported as a negative result.
  - Secondary point: the best dev BAcc.
- [x] Comparisons:
  - RoBERTa, DeBERTa and Flan-T5 alone
  - majority-vote ensemble of the three
  - cascade
  - random deferral with the same number of deferred claims per dataset (1,000 draws)
  - oracle deferral (reference only)
- [x] Cost: the measured latency of RoBERTa for every claim, plus Flan-T5's measured latency for deferred claims. Labelled *simulated*.
- [x] Statistics: paired bootstrap, resampling within datasets; 10,000 replicates; 95% CI; seed 504.
- [x] Success on test:
  - Outcome 1: the CI of (cascade − Flan-T5) is above 0.
  - Outcome 2: the CI lower bound is above −0.5, **and** the speed-up is at least 1.25×.

**Step 3: dev analysis and freezing.** Done 2026-11-15 05:51 UTC (`python scripts/phase4.py select`):
- [x] δ curve over 101 values (`results/phase4/dev_delta_curve.csv`, `fig_dev_tradeoff.png`).
- [x] Flan-T5 dev BAcc 72.38; RoBERTa 71.78.
- [x] Chosen δ:
  - primary δ = **0.19**: dev BAcc 72.40, 14.3% deferred, 0.64 of Flan-T5's time
  - secondary δ = **0.41**: dev BAcc 73.40, 38.9% deferred, 0.90 of Flan-T5's time
- [x] Frozen in `results/phase4/selected_thresholds.json` (SHA-256 `ce2102326bc8c686…`), together with the hashes of the plan and the dev inputs. The script refuses to select again once a frozen file or test results exist.

**Step 4: test evaluation, scored once.** Done (`python scripts/phase4.py evaluate`):
- [x] All approaches, all agreed metrics, paired bootstrap CIs, random deferral (1,000 draws), oracle. Outputs:
  - `results/phase4/results_test.md`, `test_summary.csv`, `test_per_dataset.csv`, `test_statistics.json`
  - `error_*.csv`, `fig_test_vs_flan.png`, `fig_test_tradeoff.png`
- [x] **Pre-registered verdict: negative.**
  - Primary cascade: 74.51, −0.21 vs Flan-T5 (CI [−1.22, +0.82]), 1.57× faster (simulated).
  - Not better than Flan-T5. Non-inferiority at −0.5 is not shown, because the CI lower bound is −1.22.
- [x] Secondary cascade: −0.04 [−0.68, +0.59], 1.09× faster. Also neither outcome.
- [x] Cascade vs RoBERTa alone: +1.37 [+0.67, +2.10]. Cascade vs random deferral: +1.12 [+0.50, +1.75]. Majority vote vs Flan-T5: +0.52 [−0.39, +1.42], at 0.45× Flan-T5's speed.
- [x] Error analysis of the cascade (confidence bins, deferral outcomes, per dataset, cost by document length, error direction). Findings are summarised in the README.
- [x] Limitation recorded, not acted on: the −0.5 margin is smaller than the achievable CI half-width (about ±1.0), and dev and test differ in the Flan-T5 − RoBERTa gap (0.60 vs 1.57).
- [x] Post-hoc, descriptive only: on the test curve (computed after freezing), the first δ reaching Flan-T5 is 0.335 and the best is 0.37 (75.06). These were **not** used to change δ.
- Process notes, for transparency:
  - A first version of the tests read the real Phase 3 files, because of a default-argument bug in `load()`. Nothing was printed, and the test stopped before writing anything, so no cascade or threshold result was seen. Fixed before the dev analysis.
  - `evaluate` was run three times to fix the figures and report formatting. Every number file is byte-identical across the runs (checked with `cmp`).

**Step 5: end-to-end Kaggle run.** Done 2026-11-15:
- [x] `scripts/run_cascade.py` and notebook stage `"cascade"` (cell 11), tested locally on CPU with tiny models first.
- [x] Code uploaded as a new version of `minicheck-code`. Notebook version 5 was stuck in the queue and cancelled. **Version 6** ran in 57 min on a Tesla T4.
  - Same environment as Phase 3. All code hashes and the threshold hash `ce2102326bc8c686` matched.
- [x] Output saved in `results/kaggle-t4-cascade/minicheck_results/`. The Phase 3 files restored inside it are identical to `results/kaggle-t4/`; the baselines are unchanged.

**Step 6: measured vs simulated.** Done (`python scripts/phase4.py measured --results results/kaggle-t4-cascade/minicheck_results`, output `results/phase4/measured_vs_simulated.json`):
- [x] Deferral decisions and predictions are identical to the simulation for all 12,949 claims. The probabilities are identical (max diff 0.0), so BAcc is 74.51 measured and simulated.
- [x] Time:
  - measured 3,380 s (261 ms/claim), simulated 3,151 s (243 ms/claim)
  - **speed-up vs Flan-T5: 1.47× measured**, 1.57× simulated
  - the gap is session variation: the RoBERTa stage was 9% slower than in Phase 3, the Flan-T5 stage 1% slower; the decision overhead is 0.45 s in total
- [x] Memory:
  - weights of both models: 4,352 MiB = 4.25 GiB
  - peak allocated: 13,617 MiB = 13.3 GiB (ClaimVerify)
  - **no OOM fallbacks**, against the plan's expectation
  - model loading: 29 s
- [x] Verdict unchanged: **negative**. The speed criterion is met (1.47× ≥ 1.25×), but non-inferiority is not shown.
- [x] README updated with the measured results next to the simulated ones.

**Reviewed and approved by the user (2026-11-15)**: methodology kept; negative statistical outcome to be reported as is.

Also done in Phase 4:
- [x] `tests/test_phase4.py`: 6 tests on synthetic data (all pass). `requirements-analysis.txt` pins the CPU analysis environment.

## Phase 5: Error analysis, reproducibility and final preparation

**Step 21: error analysis.** `scripts/error_analysis.py` → `results/phase5/`:
- [x] Test split; positive class = supported (FP = missed error, FN = false alarm).
- [x] False positives and false negatives per model and dataset:
  - RoBERTa: FN 37.8%, FP 16.5%
  - Flan-T5: FN 28.0%, FP 22.8%
  - cascade: FN 34.3%, FP 17.7%
- [x] RoBERTa / Flan-T5 agreement:
  - both right 64.0%
  - only RoBERTa wrong 10.1%
  - only Flan-T5 wrong 6.7%
  - both wrong 19.2%
  - all three wrong 16.8%
  - ExpertQA: all three wrong on 37.2%
- [x] Confident errors: 78.2% of RoBERTa's errors are kept by the gate, and 32.6% are in its most confident bin, where Flan-T5 is no better.
- [x] Dataset differences: RoBERTa is best on Reveal and ClaimVerify, DeBERTa on ExpertQA, Flan-T5 on the other 7.
- [x] Length and cost: claims with 8+ chunks are 3.6% of claims and 31.0% of cascade time.
  - The higher error rate of single-chunk claims (32.2%) comes with a dataset mix that is 40% ExpertQA. The composition is verified; the causal effect was not tested.
- [x] Claim-type definitions (numeric, negation, complex) fixed in the script before any text was examined. 12 tests check that the rules work and that no text reaches committed files.
- [x] Claim types and failure examples (`--with-text`, 2026-11-15, after the student's Hugging Face login):
  - Dataset revision `981dfd0…` (pinned). Alignment check: 12,949 texts with the same ids, labels and datasets as the predictions; no empty claims or documents.
  - Flags: numeric 28.3%, negation 9.7%, complex 38.4% of test claims.
  - Pooled results are confounded by the dataset mix (for example, complex: CNN 98%, Reveal 1.8%). So a within-dataset comparison was added, using datasets with ≥ 20 claims of each label in both groups.
  - **Complex claims (consistent):** RoBERTa is lower on complex claims in 6 of 7 datasets (mean −3.7). Flan-T5 − RoBERTa is +3.1 with the flag vs +0.3 without. The cascade defers 16.5% of complex claims vs 11.9% of the others.
  - **Numeric claims:** no consistent within-dataset effect (mixed signs over 9 datasets). The pooled advantage is a dataset-mix artefact.
  - **Negated claims:** inconclusive (4 datasets, both directions). The pooled sign reverses within datasets.
  - Failure examples: 12 (6 categories × 2 datasets). Texts are only in `private/phase5_examples.md` (git-ignored); ids are in `results/phase5/examples_idx.csv`.
  - The examples showed "supported" labels on documents without evidence. A check over all claims found the pattern rare: 58 claims (0.4%) with documents under 200 characters and a supported label, all three wrong on 43%. It does not explain ExpertQA (20 such claims).
  - The examples favour short documents (the selection rule picks the shortest), so they illustrate error types but do not estimate their frequency. This is stated in `error_analysis.md`.

**Step 22: final results.** `scripts/final_results.py` → `results/final/`:
- [x] Tables T1–T9 and figures 1–10.
- [x] 32 consistency checks recompute the numbers from the per-claim predictions and compare them with every saved table and README.md: **all PASS**.
- [x] A SHA-256 manifest of all result files.

**Step 23: repository.**
- [x] Removed byte-identical duplicates of the Phase 3 files from the Kaggle version 6 output, after comparing each file. The metrics differed only in line endings and float rounding at 1e-17.
- [x] Removed `__pycache__` folders.
- [x] Kept `results/kaggle-t4/_minicheck_results/` (the original version 1 output) for provenance, and documented it.
- [x] `.gitignore`: `private/`, parquet, model weights, caches, tokens, `.lnk`.
- [x] Secrets scan: no tokens. Text scan: no claim or document text in the results.
- [x] README rewritten for an outside reader: results at a glance, Phase 5, provenance, full reproduction order, data and privacy, citation (MiniCheck Apache-2.0, not copied). `requirements-analysis.txt` pins the CPU environment.
- [x] 19 tests pass (`python -m pytest tests -q`).
- [x] **Correction (2026-11-15, found while checking the report):** the runners record GPU memory in MiB (2^20 bytes), but README/PROGRESS had shown MiB ÷ 1,000 as "GB". Now reported in GiB: cascade peak 13,617 MiB = **13.3 GiB** (was "13.6 GB"), weights 4,352 MiB = 4.25 GiB (was "4.35 GB"), Phase 3 peaks 1.7 / 14.4 / 14.0 GiB (was 1.7 / 14.7 / 14.4). No measurement changed; the T4 limit is 15,360 MiB = 15 GiB, so all proportions are unchanged.
- [x] Final tables now keep 4 decimals, so later 2-decimal rounding cannot differ from the exact value (the cascade difference is −0.2052, shown as −0.21; a 3-decimal copy (−0.205) would round to −0.20).
- [x] Removed the short example-text fragments that had been quoted in README.md and the checklist. No dataset text remains outside `private/` (checked by search).
- [x] `.gitattributes` keeps result files byte-exact, and the manifest is written with LF endings.
- [x] Git dry run on a copy: 290 files staged; `private/` and `.lnk` ignored; no sensitive paths. On a fresh checkout with `core.autocrlf=true`, all 242 result files match the manifest, 19/19 tests pass and 32/32 checks pass.
- [ ] Student: `git init`, create the GitHub repository and push, then check online that `private/` is absent (`docs/SUBMISSION_CHECKLIST.md` C).

**Step 24: literature.**
- [x] 16 references checked against the ACL Anthology and publisher pages. BibTeX in `docs/references.bib`; relevance notes, the five suggested papers and a novelty statement in `docs/RELATED_WORK.md`. The novelty statement says the cascade is not a new method; the contribution is the evaluation.

**Step 25: submission preparation.**
- [x] `docs/SUBMISSION_CHECKLIST.md`:
  - deliverables from the brief
  - rubric bands mapped to our evidence
  - report sections mapped to evidence files
  - required points
  - ethics, resources
  - presentation plan
  - inconsistencies in the brief to ask the instructor about (20% vs 40% weight, 4 pages vs 4,000–5,000 words)
- [x] `docs/AI_USE.md`. The brief's text forbids GenAI unless approved, so cite the instructor's approval.
- [ ] Student: write the report and slides, put the five papers in the shared folder, name the folder `<Student ID>-<First Name>`.
