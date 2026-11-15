# Related work: verified references and their relevance

The bibliographic details of every entry were checked on 2026-11-15:
- **ACL Anthology entries** against their Anthology pages; the BibTeX in [references.bib](references.bib) is the Anthology's own export.
- **The other entries** against the publisher or proceedings pages: ACM DL DOI, PMLR, NeurIPS proceedings or arXiv, and the TMLR listing.

The relevance notes say only what each paper does and how it relates to this project.

The column **"In MiniCheck?"** says whether the MiniCheck paper itself cites or compares against the work (checked in the paper text).

## 1. Factual consistency and hallucination detection

| Reference | What it does | Relevance to this project | In MiniCheck? |
|---|---|---|---|
| Ji et al. (2023), *Survey of Hallucination in Natural Language Generation*, ACM Computing Surveys 55(12). [doi:10.1145/3571730](https://doi.org/10.1145/3571730) | Survey of hallucination across NLG tasks | Background: why grounded fact-checking of LLM output matters | – |
| Kryściński et al. (2020), *Evaluating the Factual Consistency of Abstractive Text Summarization*, EMNLP. [2020.emnlp-main.750](https://aclanthology.org/2020.emnlp-main.750/) | FactCC: a BERT classifier trained on **synthetic**, rule-based perturbations of source sentences | Early example of training a consistency checker on synthetic data, the idea MiniCheck develops with LLM-generated data | Cited (appendix) |
| Laban et al. (2022), *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization*, TACL 10. [2022.tacl-1.10](https://aclanthology.org/2022.tacl-1.10/) | Splits documents into sentences and aggregates NLI scores; benchmark reported in **balanced accuracy** | Source of two ideas MiniCheck uses: scoring parts of a document and aggregating, and BAcc with threshold conversion. Compared against in MiniCheck | Yes (baseline, metric) |
| Honovich et al. (2022), *TRUE: Re-evaluating Factual Consistency Evaluation*, NAACL. [2022.naacl-main.287](https://aclanthology.org/2022.naacl-main.287/) | Standardises 11 datasets and compares 12 metrics across tasks | Earlier effort to unify factual-consistency evaluation; LLM-AggreFact does the same for LLM output on grounding documents | Cited |
| Tang et al. (2023), *Understanding Factual Errors in Summarization: Errors, Summarizers, Datasets, Error Detectors*, ACL. [2023.acl-long.650](https://aclanthology.org/2023.acl-long.650/) | AggreFact benchmark and analysis of error detectors | Origin of AggreFact-CNN/XSum, two of the 10 LLM-AggreFact datasets; same first author as MiniCheck | Cited |
| Zha et al. (2023), *AlignScore: Evaluating Factual Consistency with A Unified Alignment Function*, ACL. [2023.acl-long.634](https://aclanthology.org/2023.acl-long.634/) | A 355M alignment model trained on 4.7M examples from 7 tasks; chunk-level scoring | Strongest earlier fine-tuned checker. MiniCheck reports outperforming it; a useful comparison point for model size and training data | Yes (baseline) |
| Min et al. (2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*, EMNLP. [2023.emnlp-main.741](https://aclanthology.org/2023.emnlp-main.741/) | Decomposes generations into atomic facts and checks each one | MiniCheck argues that decomposition into atomic facts is not necessary. Relevant to our "complex claim" error category | Cited |

## 2. MiniCheck and synthetic training data

| Reference | Relevance |
|---|---|
| Tang, Laban and Durrett (2024), *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents*, EMNLP. [2024.emnlp-main.499](https://aclanthology.org/2024.emnlp-main.499/), code at github.com/Liyan06/MiniCheck | **The reproduced paper.** Contributions: two synthetic data methods (C2D and D2C), the LLM-AggreFact benchmark (10 datasets), and small checkers reported as "comparable to GPT-4 but 400x cheaper". Uses a fixed threshold of 0.5 and **does not** tune per-dataset thresholds on the dev split (Section 4). Hardware: RTX A6000, 48 GB (Appendix E.2.1). |

## 3. Confidence-based routing and efficient inference

| Reference | What it does | Relevance |
|---|---|---|
| Geifman and El-Yaniv (2017), *Selective Classification for Deep Neural Networks*, NeurIPS 30. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500) | Abstains on low-confidence inputs, using the max softmax output ("softmax response") as confidence | Our gate (defer if \|p − 0.5\| < δ) is this kind of confidence rule: deferral in place of abstention |
| Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML (PMLR 70). [guo17a](https://proceedings.mlr.press/v70/guo17a.html) | Shows modern networks are often over-confident; proposes temperature scaling | A likely reason the gate misses confident errors (78% of RoBERTa's errors were not deferred). We did not calibrate; this is future work |
| Varshney and Baral (2022), *Model Cascading: Towards Jointly Improving Efficiency and Accuracy of NLP Systems*, EMNLP. [2022.emnlp-main.756](https://aclanthology.org/2022.emnlp-main.756/) | Cascades: small models first, harder inputs passed to larger models | **Our cascade is an instance of this established idea.** We do not claim a new method |
| Chen, Zaharia and Zou (2024), *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, TMLR. [arXiv:2305.05176](https://arxiv.org/abs/2305.05176) | LLM cascades that learn where to route queries to cut API cost | Shows routing for cost in the LLM setting; supports measuring cost, not just accuracy |

## 4. Reproducibility and evaluation methodology

| Reference | What it does | Relevance |
|---|---|---|
| Belz et al. (2021), *A Systematic Review of Reproducibility Research in Natural Language Processing*, EACL. [2021.eacl-main.29](https://aclanthology.org/2021.eacl-main.29/) | Reviews NLP reproduction studies and the terms used | Framing: our study re-runs released checkpoints on the same data (an evaluation reproduction), not a re-training |
| Koehn (2004), *Statistical Significance Tests for Machine Translation Evaluation*, EMNLP. [W04-3250](https://aclanthology.org/W04-3250/) | Bootstrap resampling for significance and confidence intervals | Basis of our paired bootstrap confidence intervals |
| Dror et al. (2018), *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*, ACL. [P18-1128](https://aclanthology.org/P18-1128/) | Protocol for choosing significance tests in NLP | Justifies a paired, non-parametric test for comparing systems on the same test claims |
| Card et al. (2020), *With Little Power Comes Great Responsibility*, EMNLP. [2020.emnlp-main.745](https://aclanthology.org/2020.emnlp-main.745/) | Many NLP comparisons are statistically underpowered; recommends power analysis | **Directly explains our negative non-inferiority result.** The −0.5 margin was smaller than the ±1.0 CI half-width the test set allows, and no power analysis was done before fixing it |

## Five key papers

These five cover the main lines of this project:
1. Laban et al. (2022), SummaC: chunk-and-aggregate NLI checking, and BAcc.
2. Zha et al. (2023), AlignScore: the strongest earlier checker MiniCheck compares against.
3. Tang et al. (2023), AggreFact: the benchmark origin, and the error-detector analysis.
4. Varshney and Baral (2022), Model Cascading: the method family of our extension.
5. Card et al. (2020), statistical power: the methodological lesson of our negative result.

They are read alongside the MiniCheck paper itself.

## Novelty statement

Confidence-based cascades and selective prediction are established techniques (Section 3). This project's own contribution is an **evaluation**, not a new method. We apply a standard confidence cascade to MiniCheck's released checkers with:
- a prespecified selection rule and success criteria
- thresholds chosen on dev only
- paired bootstrap CIs
- a random-deferral control
- an end-to-end measurement of time and GPU memory on a single T4

We do not claim this has never been done; a literature search for MiniCheck-specific cascades was not part of this project.
