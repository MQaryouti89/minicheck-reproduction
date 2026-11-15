# Consistency check

| check | result | detail |
|---|---|---|
| roberta-large dev: per-dataset BAcc = metrics/per_dataset.csv | PASS |  |
| roberta-large dev: macro BAcc 71.78 = metrics/summary.csv | PASS | 71.778 vs 71.778 |
| deberta-v3-large dev: per-dataset BAcc = metrics/per_dataset.csv | PASS |  |
| deberta-v3-large dev: macro BAcc 70.99 = metrics/summary.csv | PASS | 70.990 vs 70.990 |
| flan-t5-large dev: per-dataset BAcc = metrics/per_dataset.csv | PASS |  |
| flan-t5-large dev: macro BAcc 72.38 = metrics/summary.csv | PASS | 72.378 vs 72.378 |
| roberta-large test: per-dataset BAcc = metrics/per_dataset.csv | PASS |  |
| roberta-large test: macro BAcc 73.14 = metrics/summary.csv | PASS | 73.140 vs 73.140 |
| roberta-large test: macro BAcc = phase4/test_summary.csv | PASS |  |
| roberta-large test: total time 2361 s = metrics/summary.csv | PASS |  |
| deberta-v3-large test: per-dataset BAcc = metrics/per_dataset.csv | PASS |  |
| deberta-v3-large test: macro BAcc 72.56 = metrics/summary.csv | PASS | 72.561 vs 72.561 |
| deberta-v3-large test: macro BAcc = phase4/test_summary.csv | PASS |  |
| deberta-v3-large test: total time 3762 s = metrics/summary.csv | PASS |  |
| flan-t5-large test: per-dataset BAcc = metrics/per_dataset.csv | PASS |  |
| flan-t5-large test: macro BAcc 74.71 = metrics/summary.csv | PASS | 74.713 vs 74.713 |
| flan-t5-large test: macro BAcc = phase4/test_summary.csv | PASS |  |
| flan-t5-large test: total time 4954 s = metrics/summary.csv | PASS |  |
| cascade: BAcc recomputed = phase4/test_summary.csv | PASS | 74.507 |
| cascade: BAcc measured on Kaggle = simulated | PASS |  |
| cascade: deferral % recomputed = test_summary = measured | PASS | 13.64% |
| cascade: all 12,949 measured predictions equal the simulation | PASS |  |
| frozen delta file unchanged (sha256 ce2102326bc8c686) | PASS |  |
| phase5 per-dataset cascade BAcc = phase4/test_per_dataset.csv | PASS |  |
| measured: end-to-end seconds recomputed from cascade/test/*.csv | PASS | 3380.1 s |
| claim features: one row per test claim, same ids as the predictions | PASS |  |
| claim features / examples: no claim or document text columns | PASS |  |
| failure examples: ids, labels and datasets match the predictions | PASS |  |
| claim types: complex-claim cascade BAcc recomputed from flags + predictions | PASS | 69.14 |
| figure 5: measured stage sum = measured json | PASS |  |
| README.md contains the key final numbers | PASS |  |
