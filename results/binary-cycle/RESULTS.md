# Final MLG case-study results

## Completed work and scope

Corrected WL, P-WL, P-WL-C and λ experiments: MUTAG and PTC-MR, 100 outer folds each. New P-WL-C-Binary: MUTAG, 100/100 folds, 0 missing. **PTC-MR-Binary: NOT RUN — OUT OF SCOPE; Not evaluated (out of scope).** No baseline experiment was rerun. All 712 pre-existing result/cache files passed byte-for-byte preservation checks; original method functions are unchanged. Evidence: `results/binary-cycle/preservation_verified.json`, `verification.json`, `manifest.json`.

## MUTAG — primary dataset

| Method | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| WL | 88.19 ± 1.23 | 86.70 ± 1.46 |
| P-WL | 86.17 ± 1.20 | 84.25 ± 1.36 |
| P-WL-C | 87.77 ± 1.37 | 86.28 ± 1.47 |
| λ-selected | 87.77 ± 1.26 | 86.28 ± 1.35 |
| P-WL-C-Binary | 86.06 ± 1.16 | 84.30 ± 1.60 |

Mean ± population SD across ten repeat means, each averaging ten outer folds. Best mean accuracy: **WL**; best mean macro-F1: **WL**. Values rounded to two decimals can hide small differences. Full-precision changes are below.

| Change | Accuracy Δ (pp) | Macro-F1 Δ (pp) |
|---|---:|---:|
| WL → P-WL | -2.02632 | -2.45825 |
| P-WL → P-WL-C | +1.60526 | +2.03762 |
| P-WL-C → λ-selected | -0.00292 | -0.00751 |
| P-WL-C → P-WL-C-Binary | -1.71637 | -1.98133 |

P-WL-C added +1.60526 pp accuracy and +2.03762 pp macro-F1 over P-WL, despite remaining below WL. The saved accuracy difference rounds to +1.61 pp. Its observed effect differs between datasets.

## Binary versus magnitude — paired evidence

P-WL-C-Binary: accuracy 86.06 ± 1.16%; macro-F1 84.30 ± 1.60%. Versus P-WL-C: accuracy -1.71637 pp; macro-F1 -1.98133 pp.

- accuracy: 17 folds improved, 41 worsened, 42 tied.
- macro_f1: 18 folds improved, 44 worsened, 38 tied.

Predictions changed on 67/100 folds (114/1880 graph-prediction opportunities across repeats). Selected (h,p,trees) changed on 96/100 folds; individual changes: {'h': 83, 'p': 47, 'trees': 67}.

Paired fold accuracy differences range -16.66667 to +15.78947 pp; macro-F1 differences range -19.63636 to +18.51282 pp. Across ten paired repeat means, difference SD is 1.79695 pp for accuracy and 2.17208 pp for macro-F1. These are descriptive paired summaries, without significance claims.

Binary encoding changed predictions and reduced mean accuracy. Cycle magnitudes may carry useful information; this is an interpretation, not a causal result. Joint retuning also changed some selected parameters, so the measured difference is between nested method pipelines, not a fixed-model intervention.

## PTC-MR — supplementary completed evidence

| Method | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| WL | 62.33 ± 2.25 | 59.54 ± 2.47 |
| P-WL | 62.77 ± 1.03 | 60.04 ± 1.25 |
| P-WL-C | 62.22 ± 1.23 | 59.21 ± 1.51 |
| λ-selected | 62.49 ± 1.55 | 59.62 ± 1.87 |
| P-WL-C-Binary | Not evaluated (out of scope) | Not evaluated (out of scope) |

| Change | Accuracy Δ (pp) | Macro-F1 Δ (pp) |
|---|---:|---:|
| WL → P-WL | +0.44034 | +0.49549 |
| P-WL → P-WL-C | -0.55210 | -0.82427 |
| P-WL-C → λ-selected | +0.26471 | +0.40320 |

Cycles reduced mean accuracy by 0.55210 pp and macro-F1 by 0.82427 pp versus P-WL. P-WL has the highest mean accuracy and macro-F1 here. Method behavior differs across datasets; binary has not been tested here.

## Retained λ experiment

On both datasets, λ=0 exactly matched P-WL; λ∈{0.25,0.5,1,2} matched P-WL-C predictions and selected h/p/tree parameters on all 100 outer folds. Positive rescaling preserves ordering and therefore available axis-aligned split partitions in exact arithmetic. This explains the measured neutrality, but is not a guarantee across models, numerical implementations, or protocols. Joint λ selection can choose zero, so the λ-selected estimator need not equal P-WL-C.

Inner λ selection: MUTAG {'0.25': 95, '0.0': 5}; PTC-MR {'0.25': 72, '0.0': 28}. The smallest positive weight wins exact ties; it is not an empirically superior magnitude.

## Verification, sources, and figures

Baseline predictions: `results/full/MUTAG_folds.csv`, `results/full/PTC_MR_folds.csv`; baseline selection evidence: their `*_inner.csv`; original split definitions: `cache/folds/MUTAG-r10.json`, `cache/folds/PTC_MR-r10.json`; labels: `cache/data/{dataset}/{dataset}.zip`, member `{dataset}/{dataset}_graph_labels.txt`. Binary predictions/parameters: `results/binary-cycle/MUTAG_folds.csv`; 4,200 inner candidate means: `MUTAG_inner.csv`; aggregates: `MUTAG_repeat_means.csv`, `MUTAG_summary.csv`; paired evidence: `paired_folds.csv`, `paired_repeat_means.csv`, `comparison.json`. CSV metrics are fractions; differences marked pp are percentage points. Only `setting=selected` rows enter final comparisons.

Independent verification recomputed every saved baseline and binary score from labels and predictions, checked repeat aggregation, selection against the saved inner argmax with deterministic ties, checkpoint/CSV agreement, original folds, and baseline preservation. No persistence was regenerated: binary read the corrected v2 h=6 p=1/2 feature pickles. Existing GUDHI checks remain preserved; the new transformation needs no new filtration.

New 300 dpi figures: `results/binary-cycle/figures/MUTAG_accuracy_comparison.png`, `MUTAG_macro_f1_comparison.png`, `MUTAG_cycle_binary_comparison.png`, `MUTAG_paired_fold_differences.png`. Original `results/full/` figures are unchanged. Timing is in the binary evaluation JSON and manifest and excludes pre-existing feature construction. No speed improvement is claimed.

## Archived and preliminary work

`results/compact-cycle-columns/` is the interrupted compact-column attempt: MUTAG 100 folds, PTC-MR 39/100 (61 missing). Its cycle/extension results, including MUTAG ~88.20%, are superseded and excluded. WL/P-WL were reused only after original hash/vertex-block checks. Stage/smoke/preflight/resume-check outputs are development evidence, not final score sources. `results/MUTAG_component_audit.json` is an unversioned historical audit; the corrected and binary verification files are the relevant evidence.

## Limitations

Small, imbalanced datasets; repeats reuse graphs and folds are dependent. SD is descriptive variability, not a confidence interval. No significance test or causal mechanism is claimed. Binary was requested after observing the earlier study, so this is a follow-up experiment, not an independently preregistered benchmark. Depth 0–6 differs from the paper’s 0–10; inductive vocabularies, fixed modern RF defaults, no scaling, and source-order tie handling are documented adaptations. Edge labels are available but ignored. Binary was evaluated only on MUTAG; no secondary-dataset generalization claim is justified.

## Literature and attribution

P-WL-C is already a literature-based cycle enhancement in Rieck et al. (2019), §2.3.2; it is a core implementation in our study, not our novel invention. λ weighting and binary cycle presence are our custom adaptations. Implementing, modifying, and measuring includes negative and neutral findings; higher accuracy is not guaranteed.

1. Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., and Borgwardt, K. M. (2011). **Weisfeiler-Lehman Graph Kernels.** Journal of Machine Learning Research, 12(77), 2539–2561. [Publisher](https://www.jmlr.org/papers/v12/shervashidze11a.html); local `references/wl-paper.pdf`, `references/wl-paper.txt`.
2. Rieck, B., Bock, C., and Borgwardt, K. (2019). **A Persistent Weisfeiler-Lehman Procedure for Graph Classification.** Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 5448–5458. [Publisher](https://proceedings.mlr.press/v97/rieck19a.html); local `references/paper.pdf`, `references/paper.txt` (Algorithm 1, §2.3.2, Eqs. 9–12).
3. [BorgwardtLab/P-WL authors’ implementation](https://github.com/BorgwardtLab/P-WL). Snapshot identity and file hashes: `references/repository-head.txt`, `references/source_hashes.json`; feature and topology source retained locally. This is implementation evidence, not an extra claim of novelty.
4. [TUDataset catalogue](https://chrsmrrs.github.io/datasets/docs/datasets/) and [bibliography](https://chrsmrrs.github.io/datasets/docs/bibliography/). Dataset origins: Debnath, A. K., Lopez de Compadre, R. L., Debnath, G., Shusterman, A. J., and Hansch, C. (1991), *Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds*, J. Med. Chem. 34(2), 786–797; Helma, C., King, R. D., Kramer, S., and Srinivasan, A. (2001), *The Predictive Toxicology Challenge 2000–2001*, Bioinformatics 17, 107–108. Numerical dataset statistics in this study come from local original TU archives, not rounded online averages.
