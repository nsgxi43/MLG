# Final measured results

| Dataset | Method | Accuracy (%) | Macro-F1 (%) | Repeats |
|---|---|---:|---:|---:|
| MUTAG | Extension | 87.77 ± 1.26 | 86.28 ± 1.35 | 10 |
| MUTAG | P-WL | 86.17 ± 1.20 | 84.25 ± 1.36 | 10 |
| MUTAG | P-WL-C | 87.77 ± 1.37 | 86.28 ± 1.47 | 10 |
| MUTAG | WL | 88.19 ± 1.23 | 86.70 ± 1.46 | 10 |
| PTC_MR | Extension | 62.49 ± 1.55 | 59.62 ± 1.87 | 10 |
| PTC_MR | P-WL | 62.77 ± 1.03 | 60.04 ± 1.25 | 10 |
| PTC_MR | P-WL-C | 62.22 ± 1.23 | 59.21 ± 1.51 | 10 |
| PTC_MR | WL | 62.33 ± 2.25 | 59.54 ± 2.47 | 10 |

Accuracy and macro-F1 are percentages: mean ± population SD across 10 repeat means, each averaging ten outer folds. Error bars show repeat variability, not confidence intervals. The final rows use nested selection; fixed-depth and fixed-lambda curve maxima are not selected final results.

## Do cycle features help?

On **MUTAG**, adding cycles increased mean accuracy by **1.61 percentage points** relative to P-WL; macro-F1 changed by +2.04 points. WL scored 88.19%, P-WL 86.17%, P-WL-C 87.77%, and the inner-selected extension 87.77%. These are descriptive comparisons, without a significance claim.
On **PTC_MR**, adding cycles decreased mean accuracy by **0.55 percentage points** relative to P-WL; macro-F1 changed by -0.82 points. WL scored 62.33%, P-WL 62.77%, P-WL-C 62.22%, and the inner-selected extension 62.49%. These are descriptive comparisons, without a significance claim.

## Effect of lambda (our extension)

**MUTAG:** All positive weights had exactly identical predictions to P-WL-C on all 100 outer folds. Zero matched P-WL exactly. Joint inner selection chose lambda=0: 5 folds, lambda=0.25: 95 folds. The smallest positive weight wins exact positive-weight ties; that does not establish it as an optimal magnitude.
**PTC_MR:** All positive weights had exactly identical predictions to P-WL-C on all 100 outer folds. Zero matched P-WL exactly. Joint inner selection chose lambda=0: 28 folds, lambda=0.25: 72 folds. The smallest positive weight wins exact positive-weight ties; that does not establish it as an optimal magnitude.

A random forest partitions feature values by order. Multiplication by a positive constant preserves that order, explaining the flat positive-weight comparison when measured predictions coincide. Lambda zero removes cycle columns, changing both available information and the RF feature-sampling pool. Therefore this extension acts mainly as a choice of whether to include the cycle block. The nested extension can score below a baseline because it makes additional choices from noisy inner-fold estimates; its larger search space guarantees neither better outer accuracy nor better generalization.

## Validation and runtime

Both datasets completed 100 outer folds. The audit recomputed every saved accuracy and macro-F1 from stored predictions and test labels, confirmed fold coverage and inner-only selection, checked repeat aggregation and all eight 300 dpi figures. Lambda-zero predictions and metrics matched P-WL. GUDHI independently verified every graph's H0/H1 diagram at both p values and all depths. The scientific checks compared attribution against the unmodified authors' calculator, including ties and disconnected graphs.

MUTAG: evaluation wall time 1178.2 s with 8 workers; fresh cumulative feature construction through h=6 (3 trials): P-WL p=1: 0.356 ± 0.060 s, P-WL p=2: 0.357 ± 0.061 s, WL p=0: 0.228 ± 0.059 s. P-WL and P-WL-C use the same timed persistence pass. 100 unchanged baseline folds were reused; their earlier fit time is excluded from this invocation.
PTC_MR: evaluation wall time 2456.1 s with 8 workers; fresh cumulative feature construction through h=6 (3 trials): P-WL p=1: 0.524 ± 0.080 s, P-WL p=2: 0.521 ± 0.081 s, WL p=0: 0.334 ± 0.083 s. P-WL and P-WL-C use the same timed persistence pass. 39 unchanged baseline folds were reused; their earlier fit time is excluded from this invocation.

Feature and evaluation timing are separate. The runtime figures show cumulative label/persistence construction with SD across three fresh runs; they exclude classifier search and disk overhead. The whole-command duration and software versions are in manifest.json.

## Limitations and inconclusive experiments

These are small, imbalanced benchmark datasets (MUTAG 188 graphs; PTC-MR 344), and repetitions reuse the same graphs. Repeat SD is descriptive; the repeats/folds are not independent new datasets. No naive t-test or Wilcoxon test was applied. The positive lambda magnitude experiment is inconclusive as a mechanism for controlling influence under the required random-forest classifier: when predictions are identical it supplies no evidence for preferring one positive magnitude.

The protocol uses h=0..6 rather than the paper's 0..10, but matches its 10×10 outer / inner 5-fold structure, p={1,2}, and the dedicated repository scripts' tree grid {25,50,100}. It uses saved shared folds, graph-local hashes, train-only vocabulary fitting, no scaling and current explicitly fixed RF defaults. Original inner folds, historical dependency defaults and dataset-specific balancing choices were not recoverable. The tree grid is verified in the dedicated source scripts; the paper itself omits that grid. Our unbalanced class weights are an explicit protocol choice. These choices prevent test/validation leakage but differ from the authors' dataset-wide vocabulary and preprocessing. Edge labels are available but ignored by this node-labelled method. Vertex attribution and cycle-edge choices in tied filtrations depend on TU node and edge order, following the authors' code; they are not claimed to be invariant to arbitrary reindexing.

The authors' [paper/repository](https://github.com/BorgwardtLab/P-WL) reports accuracy: MUTAG WL 87.26±1.42, P-WL 86.10±1.37, P-WL-C 90.51±1.34%; PTC-MR WL 63.12±1.44, P-WL 63.07±1.68, P-WL-C 64.02±0.82%. This table is contextual reference only. Our results are not an exact reproduction of these figures; see NOTES.md for the source audit and deviations. An initial compact cycle vocabulary failed the source-dimensionality audit because zero cycle columns were omitted. That run was interrupted and retained under results/compact-cycle-columns; it is excluded from final cycle comparisons. The corrected representation retains the complete training node-label alphabet in both blocks. Exact input/fold/grid hashes and vertex-block equality allowed unchanged WL/P-WL fits to be reused; every cycle-enabled model was retrained. All corrected checks passed. Preliminary smoke runs and their differing outcomes are retained in results/ and LOG.md.
