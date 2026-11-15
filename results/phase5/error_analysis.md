# Phase 5 error analysis (test split)

Models: Phase 3 predictions on the Kaggle T4. Cascade: frozen δ = 0.19 (Phase 4). Positive class = supported, so FP = unsupported claim accepted (missed error), FN = supported claim rejected (false alarm).

## 1. Agreement between RoBERTa and Flan-T5

| dataset | claims | both right % | only RoBERTa wrong % | only Flan-T5 wrong % | both wrong % | all three wrong % |
|---|---:|---:|---:|---:|---:|---:|
| AggreFact-CNN | 558 | 74.19 | 10.93 | 5.38 | 9.50 | 8.06 |
| AggreFact-XSum | 558 | 60.22 | 14.16 | 10.39 | 15.23 | 11.83 |
| TofuEval-MediaS | 726 | 67.36 | 11.29 | 7.85 | 13.50 | 11.02 |
| TofuEval-MeetB | 772 | 72.41 | 11.40 | 6.35 | 9.84 | 7.90 |
| Wice | 358 | 68.16 | 10.89 | 8.66 | 12.29 | 10.06 |
| Reveal | 1710 | 79.77 | 3.68 | 6.55 | 10.00 | 7.84 |
| ClaimVerify | 1088 | 72.33 | 7.08 | 7.35 | 13.24 | 10.85 |
| FactCheck-GPT | 1566 | 74.20 | 5.62 | 7.85 | 12.32 | 10.09 |
| ExpertQA | 3702 | 39.68 | 14.91 | 5.56 | 39.84 | 37.17 |
| Lfqa | 1911 | 76.71 | 9.26 | 6.65 | 7.38 | 5.34 |
| All | 12949 | 64.02 | 10.09 | 6.74 | 19.15 | 16.80 |

## 2. False positives and false negatives (All = pooled)

| model | FP (missed error) | FN (false alarm) | FP rate % (of unsupported) | FN rate % (of supported) |
|---|---:|---:|---:|---:|
| RoBERTa | 863 | 2923 | 16.54 | 37.81 |
| DeBERTa | 921 | 2958 | 17.65 | 38.27 |
| Flan-T5 | 1192 | 2161 | 22.84 | 27.96 |
| Cascade | 925 | 2650 | 17.72 | 34.28 |

## 3. RoBERTa errors by confidence: what the gate can and cannot reach

| |p_RoBERTa - 0.5| | claims | roberta_errors | flan_would_fix | of_which_FP | of_which_FN | share of all RoBERTa errors % | RoBERTa error rate % |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0, 0.19) deferred | 1766 | 827 | 474 | 230 | 597 | 21.84 | 46.83 |
| [0.19, 0.30) | 1250 | 493 | 264 | 127 | 366 | 13.02 | 39.44 |
| [0.30, 0.40) | 1765 | 645 | 260 | 126 | 519 | 17.04 | 36.54 |
| [0.40, 0.45) | 1704 | 587 | 154 | 132 | 455 | 15.50 | 34.45 |
| [0.45, 0.50] | 6464 | 1234 | 154 | 248 | 986 | 32.59 | 19.09 |

## 4. Document length: errors (%) and cost

| document length (RoBERTa chunks) | claims | roberta_error | flan_error | cascade_error | deferred | roberta_ms | flan_ms | cascade_s | flan_s | share of claims % | share of cascade time % | share of Flan-T5 time % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 chunk | 7779 | 32.19 | 29.48 | 30.87 | 12.87 | 53.43 | 94.53 | 517.04 | 735.32 | 60.07 | 16.41 | 14.84 |
| 2-3 | 3726 | 23.35 | 18.44 | 20.75 | 14.87 | 223.94 | 422.50 | 1077.46 | 1574.23 | 28.77 | 34.19 | 31.78 |
| 4-7 | 983 | 27.06 | 23.70 | 26.14 | 13.12 | 457.52 | 948.42 | 579.52 | 932.29 | 7.59 | 18.39 | 18.82 |
| 8+ | 461 | 31.67 | 30.37 | 31.24 | 17.79 | 1434.18 | 3713.62 | 977.28 | 1711.98 | 3.56 | 31.01 | 34.56 |

## 5. Per-dataset BAcc (pooled within the dataset) and deferral

| dataset | RoBERTa | DeBERTa | Flan-T5 | Cascade | deferred % | best single model |
|---|---:|---:|---:|---:|---:|---|
| AggreFact-CNN | 63.75 | 64.32 | 69.95 | 69.53 | 19.35 | Flan-T5 |
| AggreFact-XSum | 70.76 | 71.05 | 74.26 | 74.43 | 14.34 | Flan-T5 |
| TofuEval-MediaS | 71.93 | 69.34 | 73.58 | 73.14 | 10.88 | Flan-T5 |
| TofuEval-MeetB | 75.94 | 72.70 | 77.30 | 75.78 | 14.64 | Flan-T5 |
| Wice | 67.57 | 69.44 | 72.42 | 67.82 | 10.89 | Flan-T5 |
| Reveal | 88.81 | 87.26 | 86.25 | 88.81 | 8.07 | RoBERTa |
| ClaimVerify | 77.58 | 75.60 | 74.59 | 77.36 | 13.33 | RoBERTa |
| FactCheck-GPT | 73.28 | 73.04 | 74.72 | 74.68 | 8.75 | Flan-T5 |
| ExpertQA | 57.43 | 58.93 | 58.87 | 57.23 | 17.59 | DeBERTa |
| Lfqa | 84.36 | 83.93 | 85.19 | 86.29 | 14.44 | Flan-T5 |

## 6. Claim types (heuristic flags; definitions in scripts/error_analysis.py)

BAcc pooled over the claims of the group; 95% CI from 1,000 resamples within datasets. Groups differ in dataset mix, so differences between groups are associations, not causes.

| claim type | present | claims | share % | RoBERTa BAcc | Flan-T5 BAcc | Cascade BAcc | Flan-T5 - RoBERTa BAcc | RoBERTa CI | Flan-T5 CI | Cascade CI | Flan-T5 - RoBERTa CI | deferred % |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| numeric | True | 3661 | 28.27 | 75.95 | 76.49 | 77.07 | 0.53 | [74.8, 77.3] | [75.1, 77.9] | [75.8, 78.3] | [-0.8, 1.8] | 12.26 |
| numeric | False | 9288 | 71.73 | 71.46 | 73.62 | 72.63 | 2.16 | [70.6, 72.4] | [72.6, 74.5] | [71.7, 73.5] | [1.3, 3.0] | 14.18 |
| negation | True | 1251 | 9.66 | 72.54 | 71.36 | 72.18 | -1.18 | [70.4, 74.7] | [68.9, 73.9] | [69.9, 74.4] | [-3.6, 1.2] | 14.07 |
| negation | False | 11698 | 90.34 | 72.88 | 74.94 | 74.20 | 2.06 | [72.2, 73.6] | [74.2, 75.8] | [73.5, 75.0] | [1.3, 2.8] | 13.59 |
| complex | True | 4976 | 38.43 | 67.47 | 69.78 | 69.14 | 2.32 | [66.2, 68.8] | [68.4, 71.2] | [67.8, 70.4] | [1.0, 3.6] | 16.48 |
| complex | False | 7973 | 61.57 | 77.47 | 78.05 | 78.20 | 0.58 | [76.5, 78.4] | [77.2, 78.9] | [77.3, 79.1] | [-0.2, 1.5] | 11.87 |

Share of claims with each flag, per dataset (%):

| dataset | numeric | negation | complex | claim words (median) |
|---|---:|---:|---:|---:|
| AggreFact-CNN | 82.1 | 19.0 | 98.0 | 54.0 |
| AggreFact-XSum | 45.0 | 5.9 | 32.3 | 22.0 |
| TofuEval-MediaS | 17.6 | 14.5 | 31.1 | 18.0 |
| TofuEval-MeetB | 43.8 | 5.3 | 38.0 | 19.0 |
| Wice | 67.9 | 7.3 | 45.5 | 22.0 |
| Reveal | 37.9 | 9.1 | 1.8 | 9.0 |
| ClaimVerify | 26.4 | 9.5 | 33.5 | 19.0 |
| FactCheck-GPT | 18.9 | 4.5 | 4.7 | 11.0 |
| ExpertQA | 19.6 | 9.3 | 61.2 | 25.0 |
| Lfqa | 15.0 | 13.8 | 43.5 | 21.0 |

### 6b. Within datasets (removes the dataset-mix confound)

Datasets where both groups have >= 20 supported and >= 20 unsupported claims. Differences in BAcc points; unweighted means over the eligible datasets.

| claim type | datasets | mean RoBERTa: with - without | mean Flan-T5: with - without | mean Cascade: with - without | mean Flan-T5 - RoBERTa, with flag | mean Flan-T5 - RoBERTa, without | datasets with RoBERTa: with - without < 0 | datasets with Flan-T5: with - without < 0 | datasets with Cascade: with - without < 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| numeric | 9 | -0.53 | -1.25 | -1.06 | 0.12 | 0.84 | 3 | 6 | 4 |
| negation | 4 | -1.52 | 0.31 | -2.32 | 1.84 | 0.01 | 2 | 1 | 2 |
| complex | 7 | -3.69 | -0.86 | -1.86 | 3.14 | 0.31 | 6 | 5 | 5 |

| claim type | dataset | claims with flag | claims without | RoBERTa: with - without | Flan-T5: with - without | Cascade: with - without | Flan-T5 - RoBERTa, with flag | Flan-T5 - RoBERTa, without |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| numeric | AggreFact-XSum | 251 | 307 | 6.65 | 1.57 | 4.22 | 1.37 | 6.45 |
| numeric | TofuEval-MediaS | 128 | 598 | 5.25 | 10.72 | 0.93 | 6.30 | 0.83 |
| numeric | TofuEval-MeetB | 338 | 434 | -16.11 | -12.33 | -12.79 | 3.50 | -0.28 |
| numeric | Wice | 243 | 115 | -10.89 | -0.34 | -7.55 | 8.81 | -1.75 |
| numeric | Reveal | 648 | 1062 | 5.36 | -2.60 | 4.03 | -8.99 | -1.03 |
| numeric | ClaimVerify | 287 | 801 | 2.27 | -4.37 | 1.76 | -7.95 | -1.31 |
| numeric | FactCheck-GPT | 296 | 1270 | -1.30 | -3.82 | -3.14 | -0.74 | 1.78 |
| numeric | ExpertQA | 726 | 2976 | 0.12 | -0.35 | -0.63 | 1.01 | 1.48 |
| numeric | Lfqa | 286 | 1625 | 3.93 | 0.28 | 3.62 | -2.26 | 1.39 |
| negation | ClaimVerify | 103 | 985 | 12.32 | 10.10 | 6.29 | -4.78 | -2.55 |
| negation | FactCheck-GPT | 71 | 1495 | -15.64 | -11.09 | -14.83 | 5.66 | 1.11 |
| negation | ExpertQA | 346 | 3356 | 1.75 | 1.66 | 1.46 | 1.34 | 1.44 |
| negation | Lfqa | 264 | 1647 | -4.50 | 0.58 | -2.20 | 5.14 | 0.06 |
| complex | AggreFact-XSum | 180 | 378 | -3.43 | -3.76 | -2.43 | 2.97 | 3.30 |
| complex | TofuEval-MediaS | 226 | 500 | -6.88 | -2.65 | -6.24 | 4.68 | 0.44 |
| complex | TofuEval-MeetB | 293 | 479 | -5.14 | -5.26 | 0.19 | 1.08 | 1.20 |
| complex | Wice | 163 | 195 | -5.22 | 6.08 | -4.79 | 11.30 | 0.00 |
| complex | ClaimVerify | 365 | 723 | 2.14 | 2.06 | 3.52 | -2.56 | -2.49 |
| complex | ExpertQA | 2266 | 1436 | -4.29 | -1.68 | -1.98 | 2.46 | -0.15 |
| complex | Lfqa | 832 | 1079 | -3.02 | -0.80 | -1.30 | 2.07 | -0.15 |

## 6c. Very short documents labelled supported (missing evidence?)

Seen in the failure examples; checked here on all test claims (document length only, no text).

| dataset | docs < 200 chars | of which supported | all three wrong on those % | all three wrong on the rest % |
|---|---:|---:|---:|---:|
| AggreFact-CNN | 0 | 0 | – | 8.06 |
| AggreFact-XSum | 4 | 3 | 100.00 | 11.35 |
| TofuEval-MediaS | 0 | 0 | – | 11.02 |
| TofuEval-MeetB | 0 | 0 | – | 7.90 |
| Wice | 0 | 0 | – | 10.06 |
| Reveal | 159 | 27 | 0.00 | 7.96 |
| ClaimVerify | 17 | 3 | 66.67 | 10.69 |
| FactCheck-GPT | 77 | 5 | 20.00 | 10.06 |
| ExpertQA | 26 | 20 | 95.00 | 36.85 |
| Lfqa | 0 | 0 | – | 5.34 |
| All | 283 | 58 | 43.10 | 16.69 |

## 7. Failure examples (ids only; text in private/phase5_examples.md, not committed)

Selected deterministically: in each category, the shortest documents (readable), one per dataset. This favours very short documents, so the examples illustrate error types; they are not a random sample and do not show how often each type occurs.

| category | idx | dataset | label | p_RoBERTa | p_DeBERTa | p_Flan-T5 | deferred | numeric | negation | complex |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Confident RoBERTa error, kept (not deferred); Flan-T5 would be right | test-03967 | Reveal | 0 | 0.973 | 0.961 | 0.028 | False | False | False | False |
| A. Confident RoBERTa error, kept (not deferred); Flan-T5 would be right | test-06576 | FactCheck-GPT | 1 | 0.025 | 0.020 | 0.618 | False | False | False | True |
| B. Deferred and broken: RoBERTa right, Flan-T5 wrong | test-04444 | Reveal | 0 | 0.469 | 0.059 | 0.573 | True | False | False | False |
| B. Deferred and broken: RoBERTa right, Flan-T5 wrong | test-06018 | FactCheck-GPT | 0 | 0.497 | 0.042 | 0.775 | True | False | False | False |
| C. All models wrong with high confidence | test-07464 | ExpertQA | 1 | 0.014 | 0.017 | 0.023 | False | False | False | True |
| C. All models wrong with high confidence | test-04093 | Reveal | 0 | 0.989 | 0.972 | 0.966 | False | False | False | False |
| D. Numeric claim, cascade wrong | test-07956 | ExpertQA | 1 | 0.010 | 0.013 | 0.280 | False | True | False | True |
| D. Numeric claim, cascade wrong | test-07294 | FactCheck-GPT | 0 | 0.535 | 0.912 | 0.945 | True | True | False | False |
| E. Negated claim, cascade wrong | test-03988 | Reveal | 0 | 0.328 | 0.925 | 0.924 | True | False | True | False |
| E. Negated claim, cascade wrong | test-06291 | FactCheck-GPT | 1 | 0.036 | 0.058 | 0.042 | False | False | True | False |
| F. Complex claim, cascade wrong | test-07628 | ExpertQA | 1 | 0.082 | 0.033 | 0.145 | False | False | False | True |
| F. Complex claim, cascade wrong | test-05102 | ClaimVerify | 1 | 0.004 | 0.013 | 0.027 | False | False | False | True |
