# Phase 4 results on the test split (simulated cascade)

Frozen on dev before this evaluation (`selected_thresholds.json`, 2026-11-15T05:51:21+00:00): primary delta = 0.19, secondary delta = 0.41.
Times are **simulated**: per-claim latencies measured separately for each model on the Kaggle T4 in Phase 3.

**Verdict: negative: neither success criterion met.**

## All approaches

| approach | BAcc (95% CI) | Δ vs Flan-T5 (95% CI) | % deferred | ms/claim (sim.) | total s (sim.) | speed-up vs Flan-T5 (sim.) |
|---|---|---|---:|---:|---:|---:|
| RoBERTa alone | 73.14 [71.98, 74.30] | -1.57 [-2.80, -0.34] | – | 182.32 | 2360.89 | 2.10 |
| DeBERTa alone | 72.56 [71.39, 73.74] | -2.15 [-3.35, -0.95] | – | 290.49 | 3761.53 | 1.32 |
| Flan-T5 alone | 74.71 [73.54, 75.95] | – | – | 382.56 | 4953.82 | 1.00 |
| Majority vote (3 models) | 75.23 [74.06, 76.40] | +0.52 [-0.39, +1.42] | – | 855.37 | 11076.24 | 0.45 |
| Cascade, primary (delta 0.19) | 74.51 [73.36, 75.70] | -0.21 [-1.22, +0.82] | 13.64 | 243.36 | 3151.29 | 1.57 |
| Cascade, secondary (delta 0.41) | 74.67 [73.50, 75.87] | -0.04 [-0.68, +0.59] | 38.79 | 351.41 | 4550.37 | 1.09 |
| Random deferral (same count per dataset, mean of draws) | 73.39 [72.32, 74.48] | -1.32 [-2.36, -0.27] | 13.64 | 236.77 | 3065.89 | 1.62 |
| Oracle deferral (reference) | 83.19 [82.16, 84.27] | +8.47 [+7.71, +9.28] | 10.09 | 224.50 | 2907.01 | 1.70 |

BAcc = mean of the 10 per-dataset balanced accuracies. CIs: paired bootstrap, 10,000 resamples within datasets.

## Precision, recall, F1 (macro over datasets; positive class = supported) and pooled confusion counts

| approach | precision_supported | recall_supported | f1_supported | precision_unsupported | recall_unsupported | f1_unsupported | macro_f1 | tp | fn | fp | tn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa alone | 81.51 | 69.75 | 73.80 | 60.76 | 76.53 | 66.22 | 70.01 | 4807.00 | 2923.00 | 863.00 | 4356.00 |
| DeBERTa alone | 81.71 | 67.34 | 72.35 | 59.04 | 77.78 | 64.96 | 68.65 | 4772.00 | 2958.00 | 921.00 | 4298.00 |
| Flan-T5 alone | 79.33 | 77.77 | 77.62 | 66.11 | 71.66 | 67.62 | 72.62 | 5569.00 | 2161.00 | 1192.00 | 4027.00 |
| Majority vote (3 models) | 82.71 | 73.37 | 76.47 | 63.85 | 77.09 | 68.42 | 72.44 | 5128.00 | 2602.00 | 906.00 | 4313.00 |
| Cascade, primary (delta 0.19) | 81.47 | 72.92 | 75.74 | 63.42 | 76.09 | 67.81 | 71.78 | 5080.00 | 2650.00 | 925.00 | 4294.00 |
| Cascade, secondary (delta 0.41) | 80.91 | 75.85 | 77.31 | 64.74 | 73.49 | 67.64 | 72.47 | 5425.00 | 2305.00 | 1055.00 | 4164.00 |
| Oracle deferral (reference) | 88.51 | 83.31 | 85.03 | 75.20 | 83.06 | 77.88 | 81.45 | 5906.00 | 1824.00 | 656.00 | 4563.00 |

## Paired bootstrap comparisons

| comparison | diff | CI low | CI high | share of replicates <= 0 |
|---|---:|---:|---:|---:|
| Cascade, primary (delta 0.19) - Flan-T5 alone | -0.205 | -1.216 | 0.820 | 0.655 |
| Cascade, primary (delta 0.19) - RoBERTa alone | 1.367 | 0.667 | 2.102 | 0.000 |
| Cascade, primary (delta 0.19) - Random deferral (same count per dataset, mean of draws) | 1.118 | 0.500 | 1.750 | 0.000 |
| Majority vote (3 models) - Flan-T5 alone | 0.517 | -0.391 | 1.418 | 0.138 |
| Cascade, secondary (delta 0.41) - Flan-T5 alone | -0.044 | -0.683 | 0.590 | 0.555 |

Random deferral: 1000 draws, BAcc mean 73.389, 95% of draws in [72.918, 73.829]; share of draws >= cascade: 0.0.

## Success criteria (fixed in configs/phase4.json)

```
{
  "outcome_1_better_than_flan": false,
  "outcome_2_non_inferior": false,
  "outcome_2_speedup_ok": true,
  "primary_time_vs_flan": 0.6361,
  "primary_speedup": 1.572,
  "outcome_2_comparable_and_faster": false,
  "result": "negative: neither success criterion met"
}
```

## BAcc per dataset

| dataset | RoBERTa alone | DeBERTa alone | Flan-T5 alone | Majority vote (3 models) | Cascade, primary (delta 0.19) | Cascade, secondary (delta 0.41) | Oracle deferral (reference) | Random deferral (same count per dataset, mean of draws) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AggreFact-CNN | 63.75 | 64.32 | 69.95 | 70.11 | 69.53 | 67.20 | 78.39 | 64.91 |
| AggreFact-XSum | 70.76 | 71.05 | 74.26 | 73.77 | 74.43 | 75.28 | 84.76 | 71.24 |
| TofuEval-MediaS | 71.93 | 69.34 | 73.58 | 73.57 | 73.14 | 73.26 | 82.54 | 72.11 |
| TofuEval-MeetB | 75.94 | 72.70 | 77.30 | 77.05 | 75.78 | 75.75 | 86.05 | 76.19 |
| Wice | 67.57 | 69.44 | 72.42 | 71.24 | 67.82 | 71.18 | 81.67 | 68.10 |
| Reveal | 88.81 | 87.26 | 86.25 | 88.60 | 88.81 | 87.62 | 92.26 | 88.60 |
| ClaimVerify | 77.58 | 75.60 | 74.59 | 77.28 | 77.36 | 75.85 | 83.92 | 77.19 |
| FactCheck-GPT | 73.28 | 73.04 | 74.72 | 75.26 | 74.68 | 75.29 | 81.98 | 73.39 |
| ExpertQA | 57.43 | 58.93 | 58.87 | 58.26 | 57.23 | 59.06 | 67.70 | 57.68 |
| Lfqa | 84.36 | 83.93 | 85.19 | 87.14 | 86.29 | 86.19 | 92.61 | 84.48 |
| Average | 73.14 | 72.56 | 74.71 | 75.23 | 74.51 | 74.67 | 83.19 | 73.39 |

## Error analysis of the primary cascade

### confidence bins dev

| |p_RoBERTa - 0.5| | n | roberta_acc | flan_acc | flan_fixes | flan_breaks | flan_ms | share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0.00, 0.05) | 523 | 49.52 | 62.33 | 164 | 97 | 384.31 | 3.47 |
| [0.05, 0.10) | 535 | 54.39 | 64.11 | 140 | 88 | 463.18 | 3.55 |
| [0.10, 0.15) | 590 | 50.34 | 66.78 | 175 | 78 | 496.33 | 3.91 |
| [0.15, 0.20) | 623 | 55.06 | 63.88 | 142 | 87 | 415.80 | 4.13 |
| [0.20, 0.25) | 646 | 59.44 | 66.56 | 135 | 89 | 440.90 | 4.28 |
| [0.25, 0.30) | 731 | 59.78 | 68.81 | 139 | 73 | 483.95 | 4.85 |
| [0.30, 0.35) | 844 | 62.56 | 68.84 | 146 | 93 | 370.89 | 5.60 |
| [0.35, 0.40) | 1092 | 62.00 | 68.22 | 149 | 81 | 439.31 | 7.24 |
| [0.40, 0.45) | 1926 | 66.46 | 68.59 | 186 | 145 | 418.15 | 12.77 |
| [0.45, 0.50) | 7570 | 81.61 | 81.00 | 188 | 234 | 373.33 | 50.20 |

### confidence bins test

| |p_RoBERTa - 0.5| | n | roberta_acc | flan_acc | flan_fixes | flan_breaks | flan_ms | share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0.00, 0.05) | 439 | 48.75 | 64.24 | 139 | 71 | 484.21 | 3.39 |
| [0.05, 0.10) | 459 | 52.29 | 65.80 | 130 | 68 | 434.89 | 3.54 |
| [0.10, 0.15) | 455 | 57.36 | 66.15 | 106 | 66 | 424.86 | 3.51 |
| [0.15, 0.20) | 529 | 54.63 | 63.89 | 123 | 74 | 469.49 | 4.09 |
| [0.20, 0.25) | 526 | 60.27 | 67.11 | 108 | 72 | 403.09 | 4.06 |
| [0.25, 0.30) | 608 | 61.68 | 71.88 | 132 | 70 | 524.71 | 4.70 |
| [0.30, 0.35) | 724 | 62.43 | 70.30 | 129 | 72 | 442.78 | 5.59 |
| [0.35, 0.40) | 1041 | 64.17 | 69.07 | 131 | 80 | 378.17 | 8.04 |
| [0.40, 0.45) | 1704 | 65.55 | 66.84 | 154 | 132 | 376.21 | 13.16 |
| [0.45, 0.50) | 6464 | 80.91 | 80.69 | 154 | 168 | 342.45 | 49.92 |

### deferral outcomes

| group | n | pct_of_claims |
|---|---:|---:|
| deferred: RoBERTa wrong, Flan-T5 right (fixed) | 474 | 3.66 |
| deferred: RoBERTa right, Flan-T5 wrong (broken) | 263 | 2.03 |
| deferred: both right | 676 | 5.22 |
| deferred: both wrong | 353 | 2.73 |
| kept: RoBERTa right | 8224 | 63.51 |
| kept: RoBERTa wrong, Flan-T5 would be right (missed) | 832 | 6.43 |
| kept: RoBERTa wrong, Flan-T5 also wrong | 2127 | 16.43 |

### per dataset

| dataset | claims | deferred_pct | bacc_roberta | bacc_flan | bacc_cascade | cascade_minus_flan | time_vs_flan |
|---|---:|---:|---:|---:|---:|---:|---:|
| AggreFact-CNN | 558 | 19.35 | 63.75 | 69.95 | 69.53 | -0.42 | 0.75 |
| AggreFact-XSum | 558 | 14.34 | 70.76 | 74.26 | 74.43 | 0.17 | 0.75 |
| TofuEval-MediaS | 726 | 10.88 | 71.93 | 73.58 | 73.14 | -0.44 | 0.55 |
| TofuEval-MeetB | 772 | 14.64 | 75.94 | 77.30 | 75.78 | -1.53 | 0.62 |
| Wice | 358 | 10.89 | 67.57 | 72.42 | 67.82 | -4.60 | 0.57 |
| Reveal | 1710 | 8.07 | 88.81 | 86.25 | 88.81 | 2.56 | 0.61 |
| ClaimVerify | 1088 | 13.33 | 77.58 | 74.59 | 77.36 | 2.77 | 0.53 |
| FactCheck-GPT | 1566 | 8.75 | 73.28 | 74.72 | 74.68 | -0.04 | 0.62 |
| ExpertQA | 3702 | 17.59 | 57.43 | 58.87 | 57.23 | -1.64 | 0.71 |
| Lfqa | 1911 | 14.44 | 84.36 | 85.19 | 86.29 | 1.11 | 0.88 |

### cost by length

| document length (RoBERTa chunks) | n | deferred_pct | cascade_s | flan_s | share_of_claims_pct | share_of_cascade_time_pct |
|---|---:|---:|---:|---:|---:|---:|
| 1 chunk | 7779 | 12.87 | 517.04 | 735.32 | 60.07 | 16.41 |
| 2-3 | 3726 | 14.87 | 1077.46 | 1574.23 | 28.77 | 34.19 |
| 4-7 | 983 | 13.12 | 579.52 | 932.29 | 7.59 | 18.39 |
| 8+ | 461 | 17.79 | 977.28 | 1711.98 | 3.56 | 31.01 |

### error direction

| approach | missed_errors_FP | false_alarms_FN |
|---|---:|---:|
| RoBERTa alone | 863 | 2923 |
| DeBERTa alone | 921 | 2958 |
| Flan-T5 alone | 1192 | 2161 |
| Majority vote (3 models) | 906 | 2602 |
| Cascade, primary (delta 0.19) | 925 | 2650 |
| Cascade, secondary (delta 0.41) | 1055 | 2305 |
| Oracle deferral (reference) | 656 | 1824 |
