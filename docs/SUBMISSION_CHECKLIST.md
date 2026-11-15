# Submission checklist (AI504 Assignment 2)

Source: the assignment brief (V2 docx, 14 Nov; the rubric is taken from V1 PDF page 8, because the V2 rubric is an image). Tick each item before submitting.

## A. Deliverables (from the brief)

- [ ] **One PDF report** submitted through Blackboard by the deadline (Week 10)
- [ ] **Length: 4,000–5,000 words including references.** The brief also says "around 4 pages", which contradicts this; ask the instructor which applies
- [ ] **Style: a research-paper format**, e.g. the ACL template (two columns)
- [ ] **References: follow Mendeley** (APA or IEEE, consistent). Import [references.bib](references.bib) into Mendeley
- [ ] **GitHub link in the report:** the repository with the code and instructions to run it (section C)
- [ ] **Shared folder** (Google Drive or Dropbox) named **`<Student ID>-<First Name>`**, containing:
  - [ ] report (PDF)
  - [ ] presentation slides
  - [ ] **five significant papers** cited in the report (PDFs); suggested five in [RELATED_WORK.md](RELATED_WORK.md), plus the MiniCheck paper
- [ ] **Presentation:** Week 11, on the day of the lecture
- [x] **AI use.** The instructor approved AI use, and the brief does not require a declaration, so at the student's request the report has no AI-use section; [AI_USE.md](AI_USE.md) is the record. Keep the written approval. The brief's text forbids GenAI unless approved
- [ ] Ask the instructor about inconsistencies in the brief:
  - weight: "20% of overall grade" on page 1 vs "40%" in the rubric
  - length: 4 pages vs 4,000–5,000 words

## B. Rubric (V1 PDF): what the top bands ask for

| Criterion (weight) | 75 (Very good) | 100 (Outstanding) | Our evidence |
|---|---|---|---|
| Interpretation (20%) | Correct research and design methodologies | Excellent demonstration of research and design methodologies | Prespecified plan (`configs/phase4.json`), dev/test separation, frozen δ with hash, controls, paired bootstrap, measured vs simulated |
| Project code (30%) | Anyone can use the repository to easily reproduce the results | Anyone can easily reproduce everything; well-annotated code, README files and scripts to aid reproduction, comments on best practices and findings | Pinned versions, one Kaggle notebook, resumable runners, 19 unit tests, 32-check consistency checker, README run instructions |
| Report (50%) | Good literature review; **at least one original contribution** to reproduce the work and/or go beyond the original results, **or error analysis**; well structured; results close to published | Very good literature discussion; goes above and beyond in analysis; effective tables and figures; results very good | Reproduction within 0.03 of the paper; 72.7 discrepancy found; cascade study with an honest negative result; error analysis; 18 verified references |

## C. GitHub repository (do before submitting)

1. [x] Repository published and **public**: https://github.com/MQaryouti89/minicheck-reproduction (verified from a fresh clone: 19/19 tests, 32/32 checks, all 242 result files match the manifest; no `private/`, tokens or dataset text).
2. [x] In the project folder:
   ```bash
   git init
   git add .
   git status            # check: no private/, no *.lnk, no tokens, no parquet, no model weights
   git commit -m "MiniCheck reproduction and adaptive cascade study"
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```
3. [x] Check online that `private/` is **not** in the repository. It holds gated dataset text. (Checked 2026-11-15: absent.)
4. [ ] Put the repository link in the report (Methodology section) and in README.md if wanted.
5. [ ] Optional: choose a licence for your own code (e.g. MIT). MiniCheck itself is Apache-2.0 and is installed from its repository, not copied.

## D. Report sections: brief requirement → evidence in this repository

| Section (brief word count) | Must cover (brief) | Evidence / where to find it |
|---|---|---|
| Title (5–10 words) | Informative, keywords | – |
| Abstract (200–250) | Objective, question, method, key findings, implications | Numbers: README "Status" and Phase 4 sections |
| Introduction (400–600) | Task, motivation, limits of prior work, proposed approach, challenges and mitigations, outline | Challenges met: gated data, T4 memory (OOM, `README` deviation note), Kaggle queue/sessions, the 72.7 discrepancy |
| Literature review (800–1,200) | Related work, gaps, theories, how the work differs | [RELATED_WORK.md](RELATED_WORK.md), [references.bib](references.bib) |
| Methodology (800–1,000) | Design, data sources and access, splits, preprocessing, GitHub link, ethics and privacy | `configs/experiment.json` (pinned versions); README "Pinned versions", "Scope", "deviation"; `configs/phase4.json`; ethics in section E below |
| Experiments and results (800–1,000) | Setup, metrics, results, tables and figures, significance | `results/final/` (tables T1–T9, figures 1–10; index in `results/final/README.md`) |
| Discussion (800–1,200) | Interpretation, issues, implications, comparison with the paper, sensitivity, limitations, **resources**, **error analysis with 2–3 failure instances** | `results/phase5/error_analysis.md` (sections 1–7); 12 examples in `private/phase5_examples.md` (local only); within-dataset claim types (T9, fig10); resources in section F below |
| Conclusion (200–250) | Findings, contribution, future work | – |
| Appendices | Extra tables, setup details | `results/final/tables.md`, `results/final/consistency_check.md` |
| Bibliography | Consistent style, Mendeley | `docs/references.bib` |

### Points the report must make clearly (from the user's brief for Phase 5)

- [ ] **Scope.** This is an *evaluation* reproduction: the released checkpoints on the released benchmark. It is not a training reproduction: the training code is not released, although the data and hyperparameters are.
- [ ] **What was reproduced.** All three checkers, on the 10 datasets, dev and test, on one Kaggle T4.
  - Test averages are within +0.03 of the paper's row means.
  - 26 of 30 dataset scores match after rounding; each of the other 4 is the size of 1–2 changed predictions.
  - Laptop and T4 give identical RoBERTa predictions.
- [ ] **What was found about the paper.** The printed RoBERTa average (72.7) does not equal the mean of its own row (73.12), unlike every other row. The benchmark is gated, and it has grown since the paper (RAGTruth etc.); the paper's 10 datasets were kept.
- [ ] **What was independently investigated.** The adaptive cascade (Phase 4), with a prespecified plan.
- [ ] **What was achieved.**
  - The measured speed-up is 1.47×.
  - The cascade beats RoBERTa (+1.37) and random deferral (+1.12).
  - Measured predictions are identical to the simulation.
  - Peak memory was 13.3 GiB (13,617 MiB) of the T4's 15 GiB, with no OOM.
- [ ] **Why the criteria were not met.**
  - The CI of (cascade − Flan-T5) is [−1.22, +0.82]: too wide to show non-inferiority at −0.5.
  - The margin was smaller than the precision the test set allows (Card et al., 2020).
  - The dev gap between Flan-T5 and RoBERTa (0.60) is smaller than the test gap (1.57), so δ under-defers on test.
  - 78% of RoBERTa's errors are confident, so the gate cannot reach them.
- [ ] **What was learned from the negative result.**
  - Confidence is informative but not sufficient.
  - A power analysis should come before fixing a margin.
  - Simulating cascades from logged predictions and latencies is exact for accuracy and within about 7% for time.
- [ ] **Limitations.**
  - Single run; the models are deterministic, so seeds are not relevant.
  - One GPU type.
  - Latencies come from different Kaggle sessions (about ±10%).
  - The claim-type flags are heuristics.
  - The test set was seen for the baselines; only the cascade threshold was kept blind.
  - The confidence is not calibrated.
- [ ] **Error analysis (2–3 instances).** Pick from `private/phase5_examples.md`, for example:
  - A: the entity mix-up between two similar names, a confident error the gate keeps
  - B: Flan-T5 using background knowledge
  - C or F: a "supported" label on a document without evidence
  Say that the examples were chosen as the shortest documents, and give the frequency check (58 claims, 0.4%).
- [ ] **Claim types.** Report the within-dataset result (complex claims) and explain why the pooled numeric and negation results are confounded.
- [ ] **Future work.**
  - calibration (temperature scaling)
  - routing on claim features, e.g. complex claims, or per-dataset/learned routing, using dev only
  - power analysis
  - other GPU types
  - deferral to an LLM

## E. Ethics and data (Methodology)

- LLM-AggreFact is **gated** on Hugging Face (accept the terms with an account). The repository contains **no document or claim text**: predictions are stored by `idx` only. The failure examples with text are kept locally in `private/` (git-ignored).
- **Before quoting example claims in the report,** check the dataset terms on its Hugging Face page. Short quotes, or paraphrases, of 2–3 claims are typical.
- No personal data was collected. The HF token was kept in Kaggle Secrets and was never written to files.
- Checker errors have consequences: the false positives are missed hallucinations (unsupported claims accepted). The cascade has fewer of these than Flan-T5 (925 vs 1,192), at the cost of more false alarms.

## F. Resources used (Discussion → Resources)

- **GPU:** Kaggle T4, about 10–11 GPU-hours in total:
  - smoke test
  - version 1: 1 h 30 min
  - version 2: failed with OOM
  - version 4: 5 h 25 min
  - version 6: about 1 h
  - see PROGRESS.md
- **Kaggle queue:** versions 3 and 5 waited in the queue and were cancelled.
- **Laptop GPU (RTX 5070 Ti):** about 20 min, for the original RoBERTa run.
- **CPU analysis:** minutes. The bootstrap with 10,000 resamples takes under 1 min.
- **People:** one student, with an AI assistant (see [AI_USE.md](AI_USE.md)).
- **Communication with the authors:** none. Optional: report the 72.7 average.

## G. Presentation checklist (Week 11)

- [ ] **Slide count:** about 10–12, for the allotted time (check with the instructor).
- [ ] **Problem and paper:** what MiniCheck claims (fig: none needed).
- [ ] **Reproduction setup:** released checkpoints, pinned versions, one T4.
- [ ] **Reproduction result:** `fig1_reproduction_vs_paper.png` and table T1. Mention the 72.7 discrepancy.
- [ ] **Extension idea:** a cascade with a confidence gate (`fig7_confidence_bins.png`).
- [ ] **Method rigour:** the prespecified plan, dev-only δ (`fig6_dev_delta_selection.png`), the frozen hash, controls.
- [ ] **Result:** `fig3_accuracy_vs_cost.png` and `fig4_difference_vs_flan_ci.png`. State the negative verdict plainly.
- [ ] **Measured vs simulated:** `fig5_simulated_vs_measured.png`.
- [ ] **Error analysis:** `fig8_error_categories.png`, `fig9_length_cost.png`, `fig10_claim_types.png`, and 1–2 short examples (quote or paraphrase within the dataset terms).
- [ ] **Lessons and limitations:** power and margin, calibration, dev/test shift.
- [ ] **Reproducibility:** GitHub, notebook, tests, consistency check.
- [ ] **Rehearse timing.** Prepare answers on: why −0.5? why δ on dev? why not tune per dataset? is 1.47× real?
