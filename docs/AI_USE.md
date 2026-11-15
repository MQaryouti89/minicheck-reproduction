# Use of generative AI in this project

**Context.** The assignment brief (V2) says: *"Generative AI should not be used for creation of the assignment and achieving the contribution."* The student reports that the instructor approved full AI use for all students in this course run. **Keep the written approval (email or announcement).** The brief does not require an AI-use declaration, and at the student's request the report has no AI-use section; this file is the record of what the AI assistant did, so the use stays transparent.

**Assistant.** Claude, an AI model by Anthropic, used through Claude Code in VS Code, in sessions of 14–15 November 2026.

## What the AI assistant did

| Area | Contribution |
|---|---|
| Phase 1, review | Read the paper (ACL Anthology, arXiv v1/v2) and the official repository. Recomputed the laptop RoBERTa result. Investigated the 72.7 vs 73.12 average discrepancy. Checked claim counts against Table 11 |
| Phases 2–3, code | Wrote `scripts/run_checker.py`, `scripts/metrics.py`, the Kaggle notebook (`scripts/build_notebook.py`), `configs/experiment.json` and `requirements-kaggle.txt`. Fixed a pandas bug in `minicheck_eval.py`. Diagnosed the CUDA out-of-memory failure and added the batch fallback. Wrote and ran local CPU tests |
| Phase 4, cascade | Proposed the plan in `configs/phase4.json` (defer rule, selection rule, margin, speed-up target, controls, bootstrap). Wrote `scripts/phase4.py`, `scripts/run_cascade.py` and the tests. Ran the dev selection and the single test evaluation. Compared the measured Kaggle run with the simulation |
| Phase 5 | Wrote `scripts/error_analysis.py` and `scripts/final_results.py` (tables, figures, consistency checks). Prepared the repository. Verified the references and wrote `docs/references.bib` and `docs/RELATED_WORK.md`. Wrote `docs/SUBMISSION_CHECKLIST.md` and this file |
| Documentation | Drafted README.md and PROGRESS.md |

## What the student did

- Chose the paper and the research direction (adaptive cascade).
- Reviewed and approved each phase, including the four prespecified Phase 4 choices: −0.5 margin, 1.25× speed-up target, a single symmetric δ, and majority vote. These were approved before the dev analysis.
- Ran every Kaggle job: the smoke test and versions 1–6. Managed Kaggle settings, secrets, inputs and uploads.
- Reviewed the Phase 3 and Phase 4 results and decided to keep the prespecified methodology and report the negative outcome.
- **Is responsible for the final report and presentation.** With the instructor's approval, the AI drafted the report text and revised it after the student's review; the student reviewed the full report, requested the corrections, and approves the final version. The presentation has not been prepared yet.

## Safeguards against AI error

- **Numbers are recomputed by code, not typed.** `scripts/final_results.py` recomputes the key numbers from the per-claim prediction files and checks them against every saved table and README.md (32 checks, all pass).
- **Thresholds were frozen before the test split was scored.** The frozen file's hash is checked by the analysis script and by the Kaggle notebook.
- **References were checked against the ACL Anthology and publisher pages.** The BibTeX comes from the Anthology's own export.
- **Problems are recorded rather than hidden** (PROGRESS.md):
  - a test bug that briefly read real files
  - three formatting re-runs of the test evaluation, with numbers byte-identical
  - a fourth, read-only re-run (`scripts/unrounded_ci.py`, into a temporary folder) to obtain unrounded CI bounds for 2-decimal reporting; all number files were byte-identical, and results/phase4 was not modified
  - the expected OOM fallbacks that did not happen
- **Suggested student checks before submission.** Re-derive two or three numbers by hand from the CSV files, for example:
  - one per-dataset BAcc
  - one confidence-bin row
  - the deferral rate
