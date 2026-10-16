# Method and protocol audit — Stage 0

Verified before implementation on 14 October 2026 against the full method and experimental sections of [Rieck et al. (2019)](https://proceedings.mlr.press/v97/rieck19a.html), the [authors' repository](https://github.com/BorgwardtLab/P-WL), and §2–3 of [Shervashidze et al. (2011)](https://www.jmlr.org/papers/v12/shervashidze11a.html). Downloaded papers and inspected source snapshots are in references/.

## Confirmed feature definitions

* WL: initial node labels at h=0; each subsequent label encodes the previous node label and sorted multiset of neighbor labels. Concatenate counts of labels for iterations 0 through h (inclusive).
* P-WL: the same refinement. At h=0, edge distance is 1 + indicator of unequal initial labels (Eq. 8). Later, use 1 + indicator of unequal previous labels + Minkowski p-distance between **neighbor-only** label-count multisets (Eq. 6, resolved by WeightAssigner in features.py). Use p=1 or 2, also as the power of persistence contributions. No recursive distance smoothing.
* Filtration: all vertices born at zero; insert only existing edges in ascending weight order. No clique expansion or triangle filling. Merge edges kill components; other edges birth essential H1 cycles.
* P-WL block: sum persistence^p by current compressed node label. Cycle block: sum birth_weight^p at **both** endpoints of each cycle-closing edge, including twice when the labels are equal. P-WL-C concatenates these blocks at each iteration, not cycle counts alone and not a persistence image.
* Repository topology.py stable-sorts edges, merges the smaller-index root into the larger-index root (despite its younger/older comments), assigns the death to the smaller root, and caps all surviving H0 components at the maximum edge weight. This resolves the paper's incomplete essential-H0 specification. We match the actual code. Edgeless graphs get a documented fallback cap of 1.

## Verified classifier and searches

The paper §3 uses random forests for WL, P-WL, P-WL-C, P-WL-UC; SVM is used for other methods outside this experiment. No SVM C is introduced here. Paper: h=0..10, p in {1,2}, 10 repetitions of 10-fold CV with inner 5-fold CV, accuracy selection.

Dedicated grid_search.py and grid_search_pwl.py search n_estimators in {25,50,100}, random_state=42, class_weight=None unless the balanced flag is supplied. Other RF options were unspecified, hence depended on historical sklearn defaults. Their inner folds are shuffled; outer seeds are repeat index 0..9. main.py instead has default 50 trees and, if requested, a different tree grid and 10 inner folds. README explicitly warns the default command is not a reproduction. Dedicated scripts standardize and then min-max scale on the entire outer training split before the inner loop; that leaks inner validation statistics.

## Decisions and deviations (predeclared)

1. Start h=0..6, as requested. The completed final range is 0..6; it was not chosen by outer accuracy. Smoke uses h=0..1 and 25 trees; full uses {25,50,100} and both p values.
2. Full = 10 repeats. Stratified shuffled outer seeds 0..9. Shared inner 5-fold seeds 1000+10*repeat+fold, persisted along with outer indices. Paper's exact original inner splits, software versions, and tree defaults could not be recovered; current RF settings are explicitly fixed: gini, sqrt max_features, bootstrap, unlimited depth, min_samples_split=2, min_samples_leaf=1, class_weight=None, seed=42.
3. Avoid dataset-wide label fitting. Graph-local deterministic SHA256 signatures permit offline caching independent of other graphs; collision registry asserts distinct signatures never collide. Fit feature vocabulary exclusively on each training split, including each inner split; unseen validation/test columns are ignored. No test graph changes training features. Hash label ordering differs from repository integer first-encounter ordering; RF randomness can be sensitive to column order.
4. No scaling: RF uses order-based splits and does not require standardization. This avoids the repository's inner leakage and preserves the meaning of lambda. Numerical rounding may still affect ties. Cache graph-local label and persistence dictionaries, not globally learned vectors.
5. Our extension is [P-WL, l * cycle] with l in {0,0.25,0.5,1,2}. At zero physically omit the cycle columns so it is exactly P-WL, including RF feature subsampling. Positive rescaling is normally invisible to RF. Do not expect a smooth lambda effect; report measured predictions, including numerical exceptions. Select lambda jointly with h, p, trees using inner accuracy only. Deterministic ties prefer smaller h, p, tree count, then lambda.
6. Fixed-depth and fixed-lambda curves are diagnostic outer evaluations, tuning other parameters internally. Never select a final method from the highest outer curve point. The four final rows are separately nested selections.
7. Use NetworkX and a small union-find persistence implementation to retain the repository's vertex attribution; validate diagrams with GUDHI independently. Initial implementation decision: parse the original TUDataset archives directly for small labelled graphs. This was subsequently corrected to invoke PyG TUDataset explicitly; see the loader correction below. Cache archives and preserve TU node IDs and first-occurrence undirected edge ordering. Edge labels/attributes are recorded but ignored, matching this node-labelled procedure.
8. Report mean and population SD of repeat means (ddof=0, matches repository). A one-repeat smoke SD is unavailable (NaN), not evidence of zero uncertainty. No significance test on overlapping folds.
9. Record graph-local feature construction separately from evaluation wall time. Feature timings measure incremental depth work shared by P-WL and P-WL-C; runtime plot shows cumulative feature construction, not invented method-specific duplicated timings.

## Reference accuracy (not measured here)

Authors' Table 1 / README, percentages: MUTAG WL 87.26±1.42, P-WL 86.10±1.37, P-WL-C 90.51±1.34; PTC-MR WL 63.12±1.44, P-WL 63.07±1.68, P-WL-C 64.02±0.82. Comparisons are contextual, not exact reproduction claims, because of the differences above.

## Implementation verification additions

The author-source attribution test executes downloaded, unmodified topology.py through a small graph adapter on random graphs including tied weights and disconnected components. GUDHI checks independent H0 death/H1 birth multisets for every dataset graph, h and p; essential H0 is capped only after comparison. The reference calculator, rather than comments about older/younger roots, defines attribution. The standard WL source explicitly distinguishes inductive from transductive feature construction (§3.1.1); this experiment uses the inductive setting.

## Computation strategy

The corrected completed full run used eight independent worker processes (earlier preflight used four) for outer folds and checkpoints completed folds with a hash of representations, labels, folds, tree grid and evaluation version. Every positive lambda setting is actually fitted in inner CV and outer evaluation. Only exact duplicate representations (Extension zero/P-WL and Extension one/P-WL-C) share fits after dictionary equality assertions. For {25,50,100} trees, a 100-tree forest supplies prefix predictions at smaller sizes; an independent fit verified exact 25-tree predictions. Deterministic seeds make these prefixes identical to separately trained smaller forests. This saves repeated fits without reducing any search range. Depth remains 0..6 for the final experiment because the complete nested joint lambda search already evaluates 77 distinct representations per inner split per outer fold; extending to 10 would increase this cost substantially. This is a deliberate computational deviation from the paper, not selection by outer scores.

## Remaining source uncertainty

The paper's experimental text specifies the classifier, h/p ranges and nested CV, but does not state a tree-count grid or all RF defaults. The {25,50,100} grid is confirmed in the dedicated repository scripts, not independently established from the paper text. The published Table 1's complete launch commands and dataset-specific balanced-flag choices could not be verified. The supplied run_grid_searches.sh invokes MUTAG without balancing, but does not cover every Table 1 run. Our class_weight=None is an explicit common choice across datasets and methods, not a claim that all paper runs used it. Source snapshots and SHA256 hashes are retained for audit; references/repository-head.txt identifies the repository HEAD observed during verification.

## Dataset loader correction during Stage 4

The initial smoke and in-progress full classifier run used a direct parser of original TU archives. To satisfy the explicit TUDataset API requirement, the delivered loader now calls torch_geometric.datasets.TUDataset (PyG 2.6.1, torch 2.8.0, cleaned=False). It seeds TUDataset's raw cache from the already downloaded identical archive, loads graph samples through the API, converts them to NetworkX, recovers original categorical node values from one-hot labels, and restores the raw TU edge order. PyG's default edge coalescing/sorting would otherwise change source-level persistence tie attribution. Per-graph equality assertions check node labels, node counts, edge sets and class labels against the direct parser used in the measurements. This is a loader/provenance correction, not a change to the graphs, folds, features or experiment. Fresh graph-local feature/persistence comparisons and the corrected smoke validation passed; evidence remains in the preserved loader/validation records and corrected smoke outputs. TUDataset's temporary one-hot alphabet is not used as the model feature vocabulary; graph-local signatures and fold-local DictVectorizer still define the representation.

## Cycle-alphabet correction during Stage 4

A late source-dimensionality check found that compact cycle dictionaries omitted zero coordinates, whereas the paper/authors' generator uses the complete node-label alphabet for both blocks. RF feature subsampling depends on the number of columns, so this difference is material. The corrected implementation retains explicit zero cycle keys for each node label. After fitting each training vocabulary, positive-lambda matrices have exactly twice as many columns as P-WL; lambda zero still omits the cycle block entirely. Both blocks remain separate before combination. Cache version was bumped to prevent stale compact vectors. The interrupted compact-column results are retained separately and excluded from all final comparisons. Corrected full results replace the earlier interim MUTAG cycle-method table.

## Verified baseline-fit reuse in the corrected run

The corrected run uses eight outer-fold workers and an optional `--reuse-base` path. To reuse a completed WL/P-WL fold, run.py reconstructs the entire former compact representation set from the retained v1 cache, proves that its vertex blocks equal the corrected v2 blocks, and evaluate.py checks the original SHA256 of labels, all representations, the full saved nested folds, tree grid and evaluation version. Only WL/P-WL inner scores and matching outer predictions are reused; all cycle-enabled fits are new. This changes computational work, not the protocol, candidate selection or baseline predictions. Corrected evaluation wall times are measured for this invocation and explicitly exclude prior baseline fits. A clean reproduction omits this optional flag and computes everything anew.

## Final case-study completion: binary cycle representation

MUTAG was selected as a small node-labeled molecular mutagenicity benchmark enabling an affordable controlled 100-fold follow-up; PTC-MR supplies a larger molecular carcinogenicity benchmark with different label balance and already-completed secondary results. Neither is chosen because an extension is guaranteed to win. Archive-verified statistics are in `results/binary-cycle/dataset_statistics.json`.

Shervashidze et al. (2011) supplies iterative neighborhood refinement and explicit subtree-pattern counts/induced kernels. Rieck et al. (2019), Algorithm 1 and §2.3.2, supplies persistent component and cycle features. We chose these papers because their shared WL refinement permits controlled topological feature comparisons. P-WL-C is an established paper-derived enhancement and a core implementation here; it is not newly invented in this project. Source-specific caps, root/edge ties and the evaluation differences above are explicit adaptations.

λ weighting was investigated to control relative cycle contribution. On both datasets, λ=0 exactly matched P-WL; λ∈{0.25,0.5,1,2} matched P-WL-C predictions and selected h/p/tree parameters on all 100 outer folds. Positive rescaling preserves ordering and therefore available axis-aligned split partitions in exact arithmetic. This explains the measured neutrality, but is not a guarantee across models, numerical implementations, or protocols. Joint λ selection can choose zero, so the λ-selected estimator need not equal P-WL-C. It remains an attempted custom modification with ineffective positive-magnitude results, not erased evidence.

Binary asks whether presence rather than magnitude is useful. `pwl.combine_binary` transforms the final aggregated cycle coordinates only. Vertex values are unchanged; all zero keys and ordering are retained. Existing corrected h=6 p=1/2 caches were read directly; no persistence, preprocessing, old CV, or PTC-MR binary computation ran. `binary_cycle.py` invokes the unchanged generic evaluator with only the new method and the original saved MUTAG folds. No globally fitted feature matrix is reused; training vocabularies remain split-local. Resume guards include input, code and environment hashes. Five regression tests cover values/zeros, dimensions/order, vertex identity, cache reuse, original source functions and checkpoint behavior; original scientific checks also passed.

**Observed outcomes**

| Change | Accuracy Δ (pp) | Macro-F1 Δ (pp) |
|---|---:|---:|
| WL → P-WL | -2.02632 | -2.45825 |
| P-WL → P-WL-C | +1.60526 | +2.03762 |
| P-WL-C → λ-selected | -0.00292 | -0.00751 |
| P-WL-C → P-WL-C-Binary | -1.71637 | -1.98133 |

PTC-MR cycle versus P-WL: −0.55210 pp accuracy / −0.82427 pp macro-F1. P-WL-C-Binary: accuracy 86.06 ± 1.16%; macro-F1 84.30 ± 1.60%. Versus P-WL-C: accuracy -1.71637 pp; macro-F1 -1.98133 pp. Predictions changed on 67/100 folds (114/1880 graph-prediction opportunities across repeats). Selected (h,p,trees) changed on 96/100 folds; individual changes: {'h': 83, 'p': 47, 'trees': 67}.

Binary encoding changed predictions and reduced mean accuracy. Cycle magnitudes may carry useful information; this is an interpretation, not a causal result. Joint retuning also changed some selected parameters, so the measured difference is between nested method pipelines, not a fixed-model intervention.

Both original datasets and the MUTAG binary follow-up are complete. PTC-MR-Binary: **NOT RUN — OUT OF SCOPE; Not evaluated (out of scope)**. Preservation verification covers 712 existing result/cache files. Small, imbalanced datasets; repeats reuse graphs and folds are dependent. SD is descriptive variability, not a confidence interval. No significance test or causal mechanism is claimed. Binary was requested after observing the earlier study, so this is a follow-up experiment, not an independently preregistered benchmark. Depth 0–6 differs from the paper’s 0–10; inductive vocabularies, fixed modern RF defaults, no scaling, and source-order tie handling are documented adaptations. Edge labels are available but ignored. Binary was evaluated only on MUTAG; no secondary-dataset generalization claim is justified.

Further work could compare an independently held-out benchmark or a different classifier sensitive to magnitude; neither was performed or selected to obtain an improvement. Complete evidence: RESULTS.md, the single 17-slide SLIDES.md, `results/binary-cycle/verification.json` and paired CSVs.
