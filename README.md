# MiniCheck reproduction and an adaptive cascade study

Paper: Liyan Tang, Philippe Laban, Greg Durrett. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.* EMNLP 2024 (Main). [ACL Anthology](https://aclanthology.org/2024.emnlp-main.499/). Code: https://github.com/Liyan06/MiniCheck (commit `b58b9fa`, Apache-2.0).

This repository has two goals:
1. **Reproduction.** Re-score the LLM-AggreFact benchmark with the three released checkers on one GPU, and compare the balanced accuracy (BAcc) on each dataset with the paper's Table 2.
2. **Extension.** Test, with a plan fixed in advance, whether a RoBERTa → Flan-T5 confidence cascade can beat Flan-T5, or match it faster.

## Results at a glance

| | Result |
|---|---|
| Reproduction (test, 12,949 claims, Kaggle T4) | RoBERTa **73.14**, DeBERTa **72.56**, Flan-T5 **74.71**; paper row means 73.12 / 72.53 / 74.70. 26 of 30 dataset scores equal the paper after rounding; the other 4 differ by 1–2 predictions |
| Issue found in the paper | Table 2 prints 72.7 as RoBERTa's average, but its own row averages 73.12 (all other rows check out) |
| Cascade (δ 0.19, frozen on dev) | 74.51 vs Flan-T5 74.71: **−0.21 [−1.22, +0.82]**; 13.6% of claims sent to Flan-T5 |
| Measured on the T4 | **1.47× faster** than Flan-T5 (3,380 s vs 4,954 s); 12,949 / 12,949 predictions identical to the simulation; peak GPU memory 13.3 GiB; no OOM |
| Prespecified verdict | **Negative.** The speed target was met, but non-inferiority (−0.5 margin) was not shown. Better than RoBERTa (+1.37) and than random deferral (+1.12) |
| Main error-analysis findings | 78% of RoBERTa's errors are confident, so a confidence gate cannot reach them. Complex claims are where RoBERTa is weak and Flan-T5 helps (+3.1 vs +0.3 points within datasets) |

- **Final tables and figures:** [results/final/](results/final/README.md).
- **Error analysis:** [results/phase5/error_analysis.md](results/phase5/error_analysis.md).
- **Related work and verified references:** [docs/RELATED_WORK.md](docs/RELATED_WORK.md), [docs/references.bib](docs/references.bib).

**Scope: evaluation reproduction.** We run the released checkpoints on the released benchmark. This checks that the published numbers come from the published models. It does not reproduce training: the training code is not released, although the 14K synthetic training examples (`lytang/C2D-and-D2C-MiniCheck`) and the hyperparameters (Appendix G) are.

Progress across all project phases: [PROGRESS.md](PROGRESS.md).

## Status

**Reproduced.** All three checkers were run on one Kaggle Tesla T4 with the same software. Their test averages are within 0.03 BAcc of the paper.

| Checker | Test BAcc (ours) | Paper: mean of Table 2 row (printed Avg) | Datasets matching | Dev BAcc | Test ms/claim | Peak GPU memory |
|---|---|---|---|---|---|---|
| MiniCheck-RoBERTa-Large | **73.14** | 73.12 (72.7) | 9/10 | 71.78 | 182 | 1.7 GiB |
| MiniCheck-DeBERTa-v3-Large | **72.56** | 72.53 (72.6) | 9/10 | 70.99 | 290 | 14.4 GiB |
| MiniCheck-Flan-T5-Large | **74.71** | 74.70 (74.7) | 8/10 | 72.38 | 383 | 14.0 GiB |

"Matching" means equal to the paper after rounding to one decimal.
- Dev and test splits each have 15,080 and 12,949 claims.
- Runtimes on the T4 (dev + test): RoBERTa 88 min, DeBERTa 140 min, Flan-T5 184 min.
- The laptop RoBERTa run (RTX 5070 Ti) gives the same 12,949 test predictions as the T4.

## Test BAcc per dataset vs Table 2 (Kaggle T4)

| Dataset | RoBERTa | paper | DeBERTa | paper | Flan-T5 | paper |
|---|---:|---:|---:|---:|---:|---:|
| AggreFact-CNN | 63.75 | 63.7 | 64.32 | 64.2 | 69.95 | 69.9 |
| AggreFact-XSum | 70.76 | 70.8 | 71.05 | 71.0 | 74.26 | 74.3 |
| TofuEval-MediaS | 71.93 | 71.9 | 69.34 | 69.3 | 73.58 | 73.6 |
| TofuEval-MeetB | 75.94 | 75.9 | 72.70 | 72.7 | 77.30 | 77.3 |
| Wice | 67.57 | 67.6 | 69.44 | 69.4 | 72.42 | 72.2 |
| Reveal | 88.81 | 88.8 | 87.26 | 87.3 | 86.25 | 86.2 |
| ClaimVerify | 77.58 | 77.4 | 75.60 | 75.6 | 74.59 | 74.6 |
| FactCheck-GPT | 73.28 | 73.3 | 73.04 | 73.0 | 74.72 | 74.7 |
| ExpertQA | 57.43 | 57.4 | 58.93 | 58.9 | 58.87 | 59.0 |
| Lfqa | 84.36 | 84.4 | 83.93 | 83.9 | 85.19 | 85.2 |
| **Average** | **73.14** | 73.12 | **72.56** | 72.53 | **74.71** | 74.70 |

26 of 30 scores match the paper after rounding. Each of the other four is the size of one changed prediction (two for Flan-T5 ExpertQA):
- RoBERTa ClaimVerify +0.18
- DeBERTa AggreFact-CNN +0.12
- Flan-T5 Wice +0.22
- Flan-T5 ExpertQA −0.13

The ranking is the paper's: Flan-T5 > RoBERTa > DeBERTa. Full metrics (precision, recall, F1, confusion matrices, latency, memory) are in `results/kaggle-t4/minicheck_results/metrics/`.

## RoBERTa result per dataset (laptop run)

| Dataset | Claims | BAcc (ours) | BAcc (paper) | Diff |
|---|---|---|---|---|
| AggreFact-CNN | 558 | 63.75 | 63.7 | +0.05 |
| AggreFact-XSum | 558 | 70.76 | 70.8 | −0.04 |
| TofuEval-MediaS | 726 | 71.93 | 71.9 | +0.03 |
| TofuEval-MeetB | 772 | 75.94 | 75.9 | +0.04 |
| Wice | 358 | 67.57 | 67.6 | −0.03 |
| Reveal | 1,710 | 88.81 | 88.8 | +0.01 |
| ClaimVerify | 1,088 | 77.58 | 77.4 | **+0.18** |
| FactCheck-GPT | 1,566 | 73.28 | 73.3 | −0.02 |
| ExpertQA | 3,702 | 57.43 | 57.4 | +0.03 |
| Lfqa | 1,911 | 84.36 | 84.4 | −0.04 |
| **Average** | 12,949 | **73.14** | 73.12 | +0.02 |

Every dataset except ClaimVerify matches the paper after rounding to one decimal. The ClaimVerify gap is the size of one changed prediction: ClaimVerify has 299 unsupported claims, so one of them flipping moves BAcc by 0.167. Seven ClaimVerify probabilities lie within 0.01 of the threshold. The Kaggle T4 run gives the same 12,949 predictions as the laptop; the largest probability difference is 5.4e-6. So the gap is not caused by the GPU. It is consistent with a small difference between our software and the authors' environment. Laptop runtime: 76 s to load, 1,041 s (about 17 min) to score all claims. T4: 2,361 s.

## Observations

- **The printed RoBERTa average in Table 2 (72.7) does not fit the paper's own aggregation.** The paper's average is the unweighted mean of the 10 dataset scores: this reproduces the printed Avg of every other row in Tables 2, 5, 7, 8 and 9 to within rounding (≤0.10). The two exceptions are both RoBERTa: Table 2 (row mean 73.12, printed 72.7) and Table 7 (73.98 vs 73.7). Weighting by dataset size gives 72.18, so that is not the explanation either. Tables 5, 8 and 9 compute their changes from 72.7. Our average (73.14) matches the per-dataset row.
- **The benchmark is gated.** `lytang/LLM-AggreFact` needs a Hugging Face login and accepting its terms. The authors' loading code does not mention this.
- **The benchmark has grown.** The test file now holds about 29K claims, including datasets added after the paper (for example RAGTruth). We keep the paper's 10 datasets. Their dev and test counts match the paper's Table 11 exactly (dev 15,080, test 12,949), and `run_checker.py` checks this on every run.
- **Training code is not released**, only the training data and hyperparameters (see Scope).

## Phase 4: adaptive RoBERTa → Flan-T5 cascade

**Status:** complete and reviewed. The simulated analysis was verified by an end-to-end run on the Kaggle T4 (see below).

**Question.** RoBERTa is 2.1× faster than Flan-T5 on the T4 but 1.57 BAcc lower. Can RoBERTa decide the claims it is confident about, and pass only uncertain ones to Flan-T5, so that the cascade:
1. beats Flan-T5, or
2. matches it at least 1.25× faster?

**Method.** The plan was fixed in [configs/phase4.json](configs/phase4.json) before any result was computed.
- **Defer rule.** A claim goes to Flan-T5 if |p_RoBERTa − 0.5| < δ, with one global δ.
- **Selecting δ, on dev only.** Primary δ: the smallest δ whose dev macro BAcc is at least Flan-T5's dev BAcc. Secondary δ: the δ with the best dev BAcc.
  - Result: **primary δ = 0.19**, **secondary δ = 0.41**.
  - Frozen in [results/phase4/selected_thresholds.json](results/phase4/selected_thresholds.json) (SHA-256 `ce2102326bc8c686…`).
- **Test.** δ is read only from that file. The single-model test results were already known from Phase 3; only the cascade settings were chosen without test data. The test evaluation code was executed four times with the same frozen file, configuration and seeds: `phase4.py evaluate` three times to fix figures and formatting, and `scripts/unrounded_ci.py` once, read-only into a temporary folder, to obtain unrounded CI bounds. All four produced byte-identical number files (PROGRESS.md, Phase 4 step 4).
- **Cost.** Each claim costs its measured Phase 3 RoBERTa latency, plus its Flan-T5 latency if deferred. So the times are **simulated**, not measured end to end.
- **Statistics.** CIs come from a paired bootstrap: 10,000 resamples within datasets, seed 504.

### Test results (simulated cost)

| Approach | Test BAcc [95% CI] | Δ vs Flan-T5 [95% CI] | % sent to Flan-T5 | ms/claim | Total (s) | Speed-up vs Flan-T5 |
|---|---|---|---:|---:|---:|---:|
| RoBERTa alone | 73.14 [71.98, 74.30] | −1.57 [−2.80, −0.34] | – | 182 | 2,361 | 2.10× |
| DeBERTa alone | 72.56 [71.39, 73.74] | −2.15 [−3.35, −0.95] | – | 290 | 3,762 | 1.32× |
| Flan-T5 alone | 74.71 [73.54, 75.95] | – | – | 383 | 4,954 | 1.00× |
| Majority vote (3 models) | 75.23 [74.06, 76.40] | +0.52 [−0.39, +1.42] | – | 855 | 11,076 | 0.45× |
| **Cascade, primary (δ 0.19)** | **74.51** [73.36, 75.70] | **−0.21 [−1.22, +0.82]** | 13.6 | 243 | 3,151 | **1.57×** |
| Cascade, secondary (δ 0.41) | 74.67 [73.50, 75.87] | −0.04 [−0.68, +0.59] | 38.8 | 351 | 4,550 | 1.09× |
| Random deferral, same count per dataset (1,000 draws) | 73.39 [72.32, 74.48] | −1.32 [−2.36, −0.27] | 13.6 | 237 | 3,066 | 1.62× |
| Oracle deferral (upper bound, reference only) | 83.19 | +8.47 | 10.1 | 225 | 2,907 | 1.70× |

**Verdict under the prespecified criteria: negative.**
- **Outcome 1 (better than Flan-T5): not met.** The CI of (cascade − Flan-T5) includes 0.
- **Outcome 2 (comparable and faster): not met.** The 1.57× speed-up passes the 1.25× target. But the CI lower bound is −1.22, below the −0.5 margin, so non-inferiority is not shown.

The primary cascade is significantly better than RoBERTa alone (+1.37 [+0.67, +2.10]) and than random deferral of the same number of claims (+1.12 [+0.50, +1.75]; none of the 1,000 random draws reached it).

Full tables are in [results/phase4/results_test.md](results/phase4/results_test.md): precision, recall and F1 for both classes, confusion counts, BAcc per dataset, and the error analysis. The figures:
- `fig_dev_tradeoff.png`: the dev δ curve used for selection
- `fig_test_vs_flan.png`: differences from Flan-T5 with their CIs; this is what the verdict uses
- `fig_test_tradeoff.png`: test point estimates. The test δ curve is drawn there for reference only. It was computed after freezing and was not used.

### Observations from the error analysis (test, primary δ)

- **RoBERTa's confidence is informative, and the pattern repeats from dev to test.** On claims with |p − 0.5| < 0.2, RoBERTa is right 53% of the time and Flan-T5 65%. In the most confident bin (|p − 0.5| ≥ 0.45, half of all claims), both are right about 81% of the time.
- **Most of RoBERTa's errors are confident ones, so a confidence gate cannot reach them.** 78% of RoBERTa's 3,786 errors are kept, not deferred. Of the 1,306 RoBERTa errors that Flan-T5 gets right, the cascade sends 474 (36%) to Flan-T5. Deferral also breaks 263 claims RoBERTa had right, for a net gain of 211.
- **One global δ fits the datasets unevenly.** Against Flan-T5 alone, the cascade gains on Reveal (+2.6) and ClaimVerify (+2.8), where RoBERTa is the stronger model. It loses most on Wice (−4.6), where Flan-T5 is much stronger but only 11% of claims are deferred.
- **Long documents dominate the cost.** Claims with 8 or more RoBERTa chunks are 3.6% of claims but 31% of the cascade's time.
- **Error direction.** The cascade misses fewer unsupported claims than Flan-T5 (925 vs 1,192 false "supported"), but raises more false alarms (2,650 vs 2,161).
- **The non-inferiority margin was too tight for this test set.** The CI half-width of the difference from Flan-T5 is about ±1.0 BAcc for the primary cascade and ±0.6 for the secondary. So a −0.5 margin could only have been met with a point estimate close to Flan-T5's or above it. The margin was fixed without a power calculation. This is a limitation of the plan, recorded here rather than changed after the fact.
- **The dev and test gaps differ.** On dev, Flan-T5 beats RoBERTa by 0.60; on test, by 1.57. A δ chosen to match Flan-T5 on dev therefore defers too little for test. The dev curve is also noisy: δ 0.19 is the first of several values that reach Flan-T5's dev BAcc.

### End-to-end run on the Kaggle T4 (measured)

The frozen cascade (δ 0.19) was run once on the 12,949 test claims (Kaggle notebook version 6, 2026-11-15, 57 min).
- **Setup.** Both checkers were loaded together. The environment was the same as Phase 3: Tesla T4, Python 3.13.16, torch 2.11.0+cu130, transformers 4.57.6. All code hashes and the threshold-file hash matched.
- **Outputs.**
  - Raw run: `results/kaggle-t4-cascade/minicheck_results/cascade/`
  - Comparison: [results/phase4/measured_vs_simulated.json](results/phase4/measured_vs_simulated.json)

| | Simulated | Measured end to end |
|---|---:|---:|
| Deferral decisions equal to the simulation | – | 12,949 / 12,949 |
| Predictions equal to the simulation | – | 12,949 / 12,949 (probabilities identical, max diff 0.0) |
| Test macro BAcc | 74.51 | **74.51** |
| % sent to Flan-T5 | 13.6 | 13.6 |
| Inference time, total | 3,151 s | **3,380 s** |
| ms per claim | 243 | **261** |
| Speed-up vs Flan-T5 alone (4,954 s, Phase 3) | 1.57× | **1.47×** |
| RoBERTa stage vs Phase 3, same claims | – | 1.09× slower |
| Flan-T5 stage vs Phase 3, same deferred claims | – | 1.01× |
| Decision overhead (end to end − stage sum) | – | 0.45 s in total |
| GPU memory of both models' weights | – | 4.25 GiB (RoBERTa 1.32 + Flan-T5 2.92) |
| Peak GPU memory allocated | – | **13.3 GiB** (13,617 MiB, ClaimVerify) of the T4's 15 GiB |
| Out-of-memory fallbacks | – | 0 |
| Model loading (one-off, not in the times above) | – | 29 s |

- **Accuracy.** The simulation is exact: the measured predictions are identical, so every accuracy result and CI above holds for the real system.
- **Time.** The measured time is 7% above the simulated time.
  - Almost all of the gap is session-to-session variation in RoBERTa's speed: the RoBERTa stage took 9% longer than in its Phase 3 session.
  - Flan-T5's stage matched its Phase 3 times.
  - Keeping both models loaded added no measurable overhead.
- **Memory.** No out-of-memory fallback was needed. The memory pressure expected in the plan did not occur, because the claims behind Flan-T5's highest memory peaks were not deferred.
- **Verdict unchanged.** The measured 1.47× still meets the ≥ 1.25× target. Non-inferiority is still not shown, because the accuracy is identical to the simulation. So the prespecified result stays **negative**.
- **Caveat.** The speed-up compares this session with Flan-T5's Phase 3 session, so it carries the same session-to-session variation (about ±10% here).

## Phase 5: error analysis (test split)

`python scripts/error_analysis.py` writes [results/phase5/](results/phase5/error_analysis.md). Positive class = supported, so a false positive (FP) is a missed error and a false negative (FN) is a false alarm.

**Verified findings**, computed from the saved predictions:
- **All three checkers err mainly by false alarms.**
  - RoBERTa rejects 37.8% of supported claims (FN), but accepts only 16.5% of unsupported ones (FP).
  - Flan-T5 is more balanced: FN 28.0%, FP 22.8%.
  - The cascade sits in between: FN 34.3%, FP 17.7%.
- **Agreement between RoBERTa and Flan-T5:**

  | Outcome | Share of claims |
  |---|---:|
  | both right | 64.0% |
  | only RoBERTa wrong | 10.1% |
  | only Flan-T5 wrong | 6.7% |
  | both wrong | 19.2% |
  | all three checkers wrong | 16.8% |

  These shared errors are the ceiling for any combination of these models.
- **ExpertQA is hard for all three checkers.** On 37.2% of its claims all three are wrong, and no checker exceeds 59 BAcc. It is also the largest test dataset (3,702 claims).
- **The gate's limit is confidence.**
  - 32.6% of RoBERTa's errors fall in its most confident bin (|p − 0.5| ≥ 0.45); there Flan-T5 is no better (both about 81% accurate).
  - Of the 1,306 RoBERTa errors that Flan-T5 gets right, only 474 lie below δ.
- **Document length drives cost, not accuracy.**
  - Single-chunk claims are 60% of claims and 16% of the cascade's time.
  - Claims with 8 or more chunks are 3.6% of claims and 31% of the time.
  - Error rates do not grow with length: 32% at 1 chunk, 23–32% for longer documents.

**Possible explanations, not tested here:**
- **ExpertQA.** Its long, expert answers may need knowledge beyond the grounding document, or its labels may be noisier.
- **Confident errors.** These are consistent with over-confident probabilities (Guo et al., 2017); calibration was not tried.
- **Short single-chunk claims.** Their higher error rate (32.2% vs 24.8% for multi-chunk) is probably a dataset-mix effect, not a length effect.
  - The composition is verified: 40% of single-chunk claims come from ExpertQA, the hardest dataset, and another 42% from Reveal and FactCheck-GPT.
  - Whether length itself matters was not tested within datasets.

**Claim types.** These were run on the pinned dataset revision with `python scripts/error_analysis.py --with-text`.
- **Alignment.** Texts, ids, labels and datasets matched the predictions for all 12,949 claims.
- **Definitions.** The flags are word rules fixed in the script before any text was examined. They cover 28.3% numeric, 9.7% negated and 38.4% complex claims.
- **Pooled vs within-dataset.** Claim types are strongly tied to datasets: 98% of AggreFact-CNN claims are complex, but only 1.8% of Reveal's. So pooled differences mix the claim type with the dataset. The comparison that matters is the one **within each dataset** (`claim_types_within_dataset*.csv`, `results/final/fig10_claim_types.png`).

**Verified findings:**
- **Complex claims are harder for RoBERTa, and this is where Flan-T5 helps.**
  - Within datasets, RoBERTa's BAcc is lower on complex claims in 6 of 7 eligible datasets (mean −3.7 points).
  - Flan-T5's lead over RoBERTa is +3.1 points on complex claims vs +0.3 on the others, positive in 6 of 7 datasets.
  - The cascade sends more complex claims to Flan-T5 (16.5% vs 11.9%).
- **Numeric claims: no consistent effect.** Pooled, numeric claims look easier (RoBERTa 76.0 vs 71.5), but within datasets the signs are mixed (9 datasets). The pooled gap comes from the dataset mix.
- **Negation: inconclusive.** Only 4 datasets have enough negated claims, and their results go both ways (−15.6 to +12.3 points for RoBERTa). The pooled pattern (Flan-T5 worse on negated claims, −1.2) turns positive within datasets (+1.8).

**Failure examples.** Two per category, from different datasets. The ids are in `results/phase5/examples_idx.csv`; the texts are in `private/phase5_examples.md` (git-ignored, gated data). The categories are:
- confident RoBERTa errors that the gate kept
- deferrals that broke a correct answer
- all three checkers confidently wrong
- numeric, negated and complex claims

They show:
- **similar names:** a claim about one person accepted against a document about a person with a similar name
- **background knowledge:** Flan-T5 accepts a fact the document does not state
- **partly supported negations**
- **"supported" labels on documents that contain no evidence:** for example, a scraped page that holds only site boilerplate

The examples were chosen as the shortest documents, so they over-represent short fragments. **The missing-evidence pattern is real but rare**:
- 58 test claims (0.4%) have a document under 200 characters and a "supported" label, and all three checkers are wrong on 43% of them (16.7% elsewhere).
- ExpertQA has only 20 such claims, so this pattern does not explain ExpertQA's 37% all-wrong rate (`short_documents.csv`).

## Pinned versions

All in [configs/experiment.json](configs/experiment.json):

| What | Version |
|---|---|
| MiniCheck code | `b58b9fa69acbd1015ec970fa65dd752413a053d2` |
| LLM-AggreFact | `981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4` (parquet files `data/dev-…`, `data/test-…`) |
| MiniCheck-RoBERTa-Large | `74c8919647e61ed0f71bc177d94f10930f090068` |
| MiniCheck-DeBERTa-v3-Large | `2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb` |
| MiniCheck-Flan-T5-Large | `96eafd01cee2d16cf81aaa2fb226b14f422a37b3` |
| Python packages (Kaggle) | [requirements-kaggle.txt](requirements-kaggle.txt): transformers 4.57.6, huggingface-hub 0.36.2, accelerate 1.15.0, nltk 3.10.4; torch is Kaggle's |

All three checkpoints are stored and run in fp32 (about 6.3 GB in total). Evaluation settings follow the authors' `benchmark_evaluation_demo.ipynb`:
- chunk size 400 tokens (RoBERTa, DeBERTa) or 500 words (Flan-T5)
- batch size 16
- the claim's score is the highest "supported" probability over the document's chunks
- supported if that score > 0.5
- BAcc per dataset, averaged without weights

**One deviation, for GPU memory.** MiniCheck splits documents at sentence boundaries, so one very long "sentence" (for example a scraped web page without punctuation) becomes a single chunk of up to 2,048 tokens. For DeBERTa and Flan-T5, 16 such chunks need more than the T4's 15 GiB; the authors used a 48 GB A6000. If a claim runs out of memory, `run_checker.py` scores that claim's chunks in batches of 8 (then 4, 2, 1). Every other claim keeps 16.

Each chunk is still scored separately, so only padding differs. A test with all three model types, on chunks of up to 1,807 tokens, gave the same probability at batch sizes 16, 8, 2 and 1: no difference for RoBERTa and DeBERTa, at most 1.4e-6 for Flan-T5. The affected claims are listed in each dataset's `.json` (`batch_fallbacks`) and counted in `metrics/summary.csv`. In the final runs this happened for 5 of 84,087 claims, each needing only one step down, to 8 chunks per batch:
- DeBERTa: one Wice dev claim and one ExpertQA test claim
- Flan-T5: three ClaimVerify test claims

None of them is in a dataset that differs from the paper.

**Provenance of the RoBERTa predictions.** They come from Kaggle version 1, which ran `run_checker.py` `326d9892…`, the code before the OOM fallback was added. DeBERTa and Flan-T5 come from version 4 (`412f0d4c…`). Function docstrings were added to all scripts after every run had finished; they change the file hashes but not the code, and `git show 1806e81:scripts/run_checker.py` gives the exact file that the recorded hash refers to.
- The scoring path is identical; the fallback only changes chunk batching after an OOM, which RoBERTa never hit.
- Version 1's original output, with its `code_version.json`, is kept in `results/kaggle-t4/_minicheck_results/`.
- Its RoBERTa predictions are byte-identical to those in `minicheck_results/`.

## Folder contents

```
README.md
PROGRESS.md                      project log for all five phases
requirements.txt                 full pinned environment of the laptop RoBERTa run
requirements-kaggle.txt          packages installed on top of Kaggle's image
requirements-analysis.txt        CPU-only analysis environment (metrics, Phase 4, tests)
configs/
  experiment.json                pinned code/data/model versions, settings, paper Table 2 and Table 11
  phase4.json                    Phase 4 analysis plan (fixed before any cascade result)
scripts/
  run_checker.py                 Kaggle runner: dev/test, per-example time, per-dataset files, resumable
  metrics.py                     CPU only: BAcc, P/R/F1, confusion matrix, runtime, GPU memory, paper comparison
  minicheck_eval.py              original laptop script that produced results/roberta-large/
  minicheck_toy.py               4 toy claims, checks that a checker loads (no gated data)
  phase4.py                      CPU only: Phase 4 select (dev) / evaluate (test, once) / measured (after Kaggle)
  run_cascade.py                 Kaggle runner for the end-to-end cascade (reuses run_checker.py's loading)
  build_notebook.py              generates notebooks/minicheck_kaggle.ipynb
  error_analysis.py              CPU only: Phase 5 error analysis (+ claim types/examples with --with-text)
  final_results.py               CPU only: final tables T1-T9, figures 1-10, 32 consistency checks, file manifest
  unrounded_ci.py                CPU only: unrounded BAcc and CIs for T3 (read-only re-run of the test evaluation)
  report_figures.py              CPU only: the report's data figures at print size (results/final/report_figures/)
  cascade_diagram.py             CPU only: the report's cascade diagram (Fig. 1), values read from saved results
tests/
  test_phase4.py                 6 tests: cascade rule, BAcc, paired bootstrap, freezing, measured vs simulated
  test_error_analysis.py         13 tests: claim-type rules, within-dataset and short-document checks, no gated text in outputs, id/label matching
notebooks/
  minicheck_kaggle.ipynb         one notebook: smoke test, RoBERTa, DeBERTa, Flan-T5, cascade, comparison
docs/
  RELATED_WORK.md, references.bib    18 verified references and their relevance
results/
  roberta-large/                 laptop run (test split only)
  kaggle-t4/minicheck_results/   Phase 3 Kaggle T4 runs (all three models, dev + test); baselines, not modified
  kaggle-t4/_minicheck_results/  original Kaggle version 1 output (RoBERTa only), kept for provenance
  phase4/                        Phase 4: frozen delta, dev curve, test results, figures, measured vs simulated
  kaggle-t4-cascade/minicheck_results/   Phase 4 Kaggle run (version 6): cascade/test/ per-claim results, log
  phase5/                        Phase 5 error analysis (CSV, figures, error_analysis.md)
  final/                         final tables, figures, consistency check, manifest (see its README)
private/                         (git-ignored) failure examples with gated text, created by --with-text
```

Kaggle runs produce `minicheck_results/` with this layout:

```
<model>/<split>/<dataset>.csv     idx,dataset,label,prob,pred,n_chunks,latency_ms   (one row per claim)
<model>/<split>/<dataset>.json    runtime, latency mean/median/p95, peak GPU memory, versions
<model>/runs.jsonl                one line per session: model load time, GPU memory of the weights, environment
metrics/per_dataset.csv           all metrics per model/split/dataset, plus macro 'Average' and 'Pooled' rows
metrics/summary.csv               one row per model/split
metrics/comparison_test.md        test BAcc vs Table 2 for all models
metrics/hardware_check.json       laptop vs T4 RoBERTa predictions
logs/, code_version.json, pip_freeze_*.txt
```

**Units.** GPU memory fields named `*_mb` in the result files are MiB (2^20 bytes); this README reports them in GiB. The T4 has 15,360 MiB (15 GiB) usable.

`idx` is `<split>-<row number in the pinned parquet file>`, for example `test-00042`. It is identical across models and runs, so predictions can be joined on it. Prediction files hold no document or claim text, because the dataset is gated.

## How to run

**Kaggle (main experiments).** Open [notebooks/minicheck_kaggle.ipynb](notebooks/minicheck_kaggle.ipynb) on Kaggle and follow its first cell. In short: GPU T4 x1, Internet on, `HF_TOKEN` secret, then one *Save & Run All* per stage (smoke → RoBERTa → DeBERTa → Flan-T5 → cascade). Before each new version, attach the previous version's output so finished results are reused.

**Any machine with a GPU:**

```bash
pip install -r requirements-kaggle.txt   # plus torch for your CUDA version
pip install --no-deps "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@b58b9fa69acbd1015ec970fa65dd752413a053d2"
export HF_TOKEN=...                      # after accepting the LLM-AggreFact terms

python scripts/run_checker.py roberta-large --out smoke --limit 5          # smoke test
python scripts/run_checker.py roberta-large --splits dev test --out results/kaggle-t4
python scripts/metrics.py results/kaggle-t4 --legacy results/roberta-large/roberta-large_preds.csv
```

Re-running the same command resumes. Finished datasets are skipped, and a partly scored dataset continues from its `.part.csv`.

`run_checker.py` calls MiniCheck's own per-example function (`Inferencer.inference_per_example`, which `MiniCheck.score()` loops over), so each claim can be timed. At start-up it checks that this gives the same probabilities as `score()` on three claims.

**Analysis (CPU only, from the saved predictions; no GPU and no gated data needed; Python 3.13, since the pinned numpy 2.5.4 does not install on 3.11):**

```bash
pip install -r requirements-analysis.txt   # numpy, pandas, scikit-learn, matplotlib, pytest (pinned)
python scripts/metrics.py results/kaggle-t4/minicheck_results --legacy results/roberta-large/roberta-large_preds.csv
python scripts/phase4.py select      # dev: delta curve, chooses and freezes delta (refuses: already frozen)
python scripts/phase4.py evaluate    # test: reads the frozen delta; refuses if the plan changed
python scripts/phase4.py measured --results results/kaggle-t4-cascade/minicheck_results
python scripts/error_analysis.py     # Phase 5 (add --with-text after `hf auth login` for claim types/examples)
python scripts/final_results.py      # final tables/figures + consistency check (exits with an error on any mismatch)
python -m pytest tests -q            # 19 tests
```

**Reproducing everything from scratch.**
1. Accept the LLM-AggreFact terms on Hugging Face.
2. On Kaggle, run the notebook stages in order: smoke → roberta-large → deberta-v3-large → flan-t5-large. This takes about 7.5 GPU-hours on a T4.
3. Run `phase4.py select` and `phase4.py evaluate`.
4. Run the notebook stage `cascade` (about 1 hour). The notebook checks the frozen threshold's hash first.
5. Run the remaining analysis commands above.

Because the checkers are deterministic, a rerun should give the same predictions. Ours were identical between a laptop RTX 5070 Ti and the T4.

**Original laptop script** (kept for the existing RoBERTa result): `python scripts/minicheck_eval.py roberta-large results/ [--limit 5]`.

## Reproducing each report table and figure

Run from the repository root. The GPU stages (per-claim predictions) run on Kaggle; everything below them runs on a CPU from the saved predictions, without the gated text unless noted.

| Report item | Saved file | Command |
|---|---|---|
| Per-claim predictions (all tables) | `results/kaggle-t4/minicheck_results/<model>/<split>/*.csv` | `python scripts/run_checker.py <model> --splits dev test --out results/kaggle-t4` (GPU) |
| Measured cascade run (Table 4) | `results/kaggle-t4-cascade/minicheck_results/cascade/test/*.csv` | `python scripts/run_cascade.py --out results/kaggle-t4-cascade/minicheck_results` (GPU) |
| Per-dataset metrics | `results/kaggle-t4/minicheck_results/metrics/` | `python scripts/metrics.py results/kaggle-t4/minicheck_results --legacy results/roberta-large/roberta-large_preds.csv` |
| Table 1: test BAcc vs the original Table 2 | `results/final/T1_reproduction_vs_paper.csv` | `python scripts/final_results.py` |
| Table 2: cost per checker | `results/final/T2_baselines_dev_test_cost.csv` | `python scripts/final_results.py` |
| Table 3: all approaches, test | `results/phase4/test_summary.csv`, `results/final/T3_all_approaches_test.csv`, `T3_unrounded.csv` | `python scripts/phase4.py evaluate`, then `python scripts/final_results.py` and `python scripts/unrounded_ci.py` |
| Table 4: simulated vs measured | `results/phase4/measured_vs_simulated.json`, `results/final/T5_simulated_vs_measured.csv` | `python scripts/phase4.py measured --results results/kaggle-t4-cascade/minicheck_results`, then `python scripts/final_results.py` |
| Table 5: failure examples (ids and probabilities) | `results/phase5/examples_idx.csv` | `python scripts/error_analysis.py --with-text` (needs the gated test file) |
| Table A1: pinned versions and settings | `configs/experiment.json`, `configs/phase4.json` | read by every script |
| Table B1: precision, recall, F1 | `results/final/T6_precision_recall_f1_test.csv` | `python scripts/final_results.py` |
| Table C1: claim types within datasets | `results/final/T9_claim_types_within_dataset.csv` | `python scripts/error_analysis.py --with-text`, then `python scripts/final_results.py` |
| Threshold selection (Fig. 3) | `results/phase4/selected_thresholds.json`, `dev_delta_curve.csv` | `python scripts/phase4.py select` (refuses to run again once δ is frozen) |
| Fig. 1: cascade diagram | `results/final/report_figures/fig_cascade_diagram.png` | `python scripts/cascade_diagram.py` |
| Figs. 2–10 | `results/final/report_figures/*.png` | `python scripts/report_figures.py` |

`python scripts/final_results.py` also re-runs the 32 consistency checks and exits with an error if any saved number differs from the value recomputed from the per-claim predictions.

## Advice for future reproducers

- **Pin the benchmark revision and check the counts.** LLM-AggreFact is gated and has grown since publication. Accept its terms, pin a revision, keep the ten original datasets and check the claim counts (dev 15,080, test 12,949) before scoring; `run_checker.py` does this on every run.
- **Expect memory limits on a 16 GB GPU.** A few documents contain one very long "sentence" that becomes a chunk of up to 2,048 tokens; at batch size 16 these exceed a T4's memory for DeBERTa and Flan-T5. Score such claims with smaller chunk batches (as `run_checker.py` does) and record which claims needed it.
- **Plan around session limits.** Run one model per Kaggle session, use `--stop-after-min` so progress is saved before the 12-hour limit, and attach the previous version's output so finished datasets are skipped. Long full-power runs on a laptop GPU are risky: ours shut down twice under sustained load.
- **Compare like with like.** Average per-dataset BAcc without weights, keep the 0.5 threshold, and compare with the per-dataset rows of the original Table 2 rather than its printed averages (the RoBERTa average is inconsistent with its row).
- **Freeze before testing.** Choose any threshold on the development split, freeze it in a file and check its hash before the test split is scored; `phase4.py` refuses to re-select once δ is frozen.
- **Time systems in the same session.** Speed differs by up to about 10% between Kaggle sessions, so speed-ups that combine sessions carry that uncertainty.
- **Keep gated text out of public files.** Store predictions by claim id only; `tests/test_error_analysis.py` checks that no claim or document text reaches the results folder.

## Data, privacy and credentials

- LLM-AggreFact is gated. **This repository contains no document or claim text**: predictions are stored by `idx` only.
- The Hugging Face token is read from Kaggle Secrets or the local Hugging Face login, and is never written to files.
- Model weights are downloaded at run time and are not stored here.

## Citation and acknowledgements

This work evaluates the released MiniCheck checkpoints, code and benchmark by Tang, Laban and Durrett.
- **Code:** MiniCheck's code (Apache-2.0) is installed from its repository at a pinned commit, not copied.
- **Settings:** the evaluation settings follow the authors' `benchmark_evaluation_demo.ipynb`.

```bibtex
@inproceedings{tang-etal-2024-minicheck,
    title = "{M}ini{C}heck: Efficient Fact-Checking of {LLM}s on Grounding Documents",
    author = "Tang, Liyan and Laban, Philippe and Durrett, Greg",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.499/",
    doi = "10.18653/v1/2024.emnlp-main.499"
}
```

Other references used in this project: [docs/references.bib](docs/references.bib).
