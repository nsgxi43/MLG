# Comprehensive MLG case-study presentation — 17 slides

This is the single faculty-ready outline. Tables are generated from audited saved predictions; see RESULTS.md and the traceability appendix below.

## Slide 1 — Title and Case Study Objective

**Core content**

**Beyond Weisfeiler-Lehman: Evaluating Persistent and Cycle-Aware Graph Representations**

Machine Learning on Graphs case study. Implement established WL representations; investigate how modifications to topological cycle features affect graph classification. Completed: WL, P-WL, P-WL-C and λ on both datasets; binary on MUTAG only.

**Suggested visualization**

Title with the five representation names; highlight the binary variant as the primary custom representation.

**Presenter notes**

Our goal is implementation, controlled modification, and measurement, not guaranteed improvement.

## Slide 2 — Problem Statement and Motivation

**Core content**

Graph classification assigns a label to an entire graph. Tabular counts can omit connectivity; labeled neighborhood patterns retain more structure. WL offers efficient explicit features and induced graph kernels, but can miss global connectivity and cycles. Research question: do persistence, cycle magnitudes, weighting, and cycle presence change classification?

**Suggested visualization**

A uniform six-cycle beside two disjoint triangles: same degree/local WL labels, different component/cycle counts.

**Presenter notes**

This counterexample illustrates a limitation, not a claim that all WL features ignore every cycle. `checks.py` validates the example.

## Slide 3 — Datasets: MUTAG and PTC-MR

**Core content**

| Dataset | Graphs / classes | Class counts (-1,+1) | Nodes / undirected edges | Atom / bond label types | Average nodes / edges |
|---|---|---|---|---|---|
| MUTAG | 188 / 2 | 63, 125 | 3371 / 3721 | 7 / 4 | 17.93085 / 19.79255 |
| PTC_MR | 344 / 2 | 192, 152 | 4915 / 5054 | 18 / 4 | 14.28779 / 14.69186 |

MUTAG: molecular mutagenicity classification, primary binary dataset. PTC-MR: molecular carcinogenicity classification for male rats, supplementary earlier-method evidence. Nodes represent atoms, edges bonds; both have categorical node/edge labels and no continuous node/edge attributes. The implemented methods use node labels and topology, ignoring edge labels. These small established benchmarks allow controlled, affordable nested comparisons involving cyclic molecular structures.

**Suggested visualization**

The dataset table plus one schematic atom/bond graph; use drawings only as illustrations, not reconstructed chemistry.

**Presenter notes**

All numeric statistics were recomputed from local original TU archives and checked against `_data.json`; raw classes are -1/+1. PTC-MR-Binary: Not evaluated (out of scope).

## Slide 4 — Research Paper Foundation

**Core content**

Shervashidze et al. (2011): efficient kernels for labeled graph classification via iterative neighborhood-label refinement and concatenated subtree-pattern counts; local refinement can miss global topology. Rieck et al. (2019): enrich WL features using filtrations and persistent connected-component/cycle information. Its Algorithm 1 and §2.3.2 define P-WL and P-WL-C; cycle inclusion is established literature, not our novel invention. Our source-following caps/ties, inductive vocabularies, h≤6, and modern RF defaults are documented choices; λ and binary are custom adaptations.

Full citations: slide 17.

**Suggested visualization**

Two paper-title cards with “neighborhood counts” → “persistent components and cycles.”

**Presenter notes**

Paper-inspired reimplementation with deviations, not exact reproduction of published scores. The original source resolves attribution details the paper leaves underspecified.

## Slide 5 — Original Weisfeiler-Lehman Method (WL)

**Core content**

`l₀(v)` is the categorical node label. `lₕ(v) = hash(lₕ₋₁(v), sorted multiset of neighbor labels)`. `φ_WL(G)` concatenates label-count coordinates for iterations 0…h. Example: a center labeled A with B,B neighbors gets a different refined signature from A with B,C neighbors. The induced kernel is `k(G,G′)=〈φ(G),φ(G′)〉`; experiments feed explicit vectors to a Random Forest, without constructing a Gram matrix.

**Suggested visualization**

Two labeled three-node neighborhoods and their distinct refined signatures, then a count vector.

**Presenter notes**

Implementation: `wl.py` signature, wl_features, cumulative. SHA256 names keep graph-local caching independent of other graphs.

## Slide 6 — Persistent Weisfeiler-Lehman (P-WL)

**Core content**

At h=0: `w(u,v)=1+1[label(u)≠label(v)]`. At h>0: add the Minkowski p-distance between previous-label neighbor-count vectors. Begin with vertices at zero and insert existing edges in stable increasing weight order. Component merges give deaths; surviving components are capped at the maximum edge weight (edgeless fallback 1). `Vₕ(a)=Σ_{v:lₕ(v)=a} death(v)^p`, p∈{1,2}; concatenate h=0…H. Unlike WL counts, vertex occurrences are persistence-weighted.

**Suggested visualization**

A four-node weighted edge-insertion sequence showing a merge and component death.

**Presenter notes**

Implementation: `pwl.edge_weights`, `pwl.persistence`, `pwl.pwl_features`. The same p controls edge distance and contribution powers. Attribution follows source index order, not an arbitrary reindexing-invariance claim.

## Slide 7 — Cycle-Feature Enhancement (P-WL-C)

**Core content**

A graph cycle is a closed route; cycle information complements merge-only H0 features. If an inserted edge connects already-connected vertices, it births H1. `Cₕ(a)=Σ_{e=(u,v) closing a cycle} w(e)^p × (1[lₕ(u)=a]+1[lₕ(v)=a])`. Same-label endpoints contribute twice. `φ_P-WL-C=[V₀…V_H, C₀…C_H]` with depth-namespaced coordinates and complete zero-inclusive cycle alphabet. No triangles are filled.

MUTAG: P-WL 86.17%, P-WL-C 87.77%; exact +1.60526 pp (+1.61 rounded); WL 88.19%. PTC-MR: P-WL 62.77%, P-WL-C 62.22%; −0.55210 pp.

**Suggested visualization**

Highlight the final cycle-closing edge and both endpoint-label bins; show `results/full/MUTAG_ablation.png`.

**Presenter notes**

This paper-derived feature enhancement is a core part of our implemented study. It helped versus P-WL on MUTAG and hurt on PTC-MR; remaining below WL does not make it irrelevant.

## Slide 8 — λ-Weighted Cycle-Feature Extension

**Core content**

`φλ=[V, λC]`, λ∈{0,0.25,0.5,1,2}. λ=0 physically removes cycle columns and equals P-WL, including RF feature sampling. Every positive tested value yielded identical predictions and selected h/p/tree parameters to P-WL-C on all 100 folds per dataset. Scaling preserves value order; axis-aligned trees often retain split partitions. The λ-selected model also chooses include/exclude, so its final scores can differ slightly.

**Suggested visualization**

`results/full/MUTAG_lambda.png` and `results/full/PTC_MR_lambda.png`.

**Presenter notes**

Keep this legitimate but empirically ineffective magnitude tweak visible. Do not call 0.25 optimal: it wins exact ties. No universal invariance guarantee.

## Slide 9 — Proposed Binary Cycle Representation (P-WL-C-Binary)

**Core content**

Hypothesis: can presence of a cycle pattern be more informative than its magnitude? `Bₕ(a)=1[Cₕ(a)≠0]`; `φ_binary=[V,B]`. Binarize the completed aggregate coordinate, not each edge. V is untouched; keys, ordering, zero-inclusive vocabulary, and dimensions are preserved. Example: `[0,2,8]→[0,1,1]`. Positive values collapse into one category, removing magnitude distinctions; this is a representation change unlike positive λ rescaling. MUTAG only.

**Suggested visualization**

Side-by-side magnitude and presence vectors; underline identical V blocks.

**Presenter notes**

Implementation: `pwl.combine_binary`. It reads existing corrected caches through `binary_cycle.read_inputs`; no new persistence or graph preprocessing.

## Slide 10 — Implementation and System Architecture

**Core content**

Dataset → preprocessing → WL refinement → persistent V/C → feature variants → training-only vocabulary → Random Forest → nested evaluation. `data.py`: original TUDataset/NetworkX preprocessing; `wl.py`: label refinement; `pwl.py`: persistence/variants; `evaluate.py`: shared splits, RF, selection, checkpoints; `binary_cycle.py`: cache-only MUTAG runner; `test_binary.py`: transformation/resume regression checks; `finalize_case_study.py`: prediction audit, comparisons, figures, reports. Cached graph-local features are transformed without globally fitting vocabularies.

**Suggested visualization**

```mermaid
flowchart LR
  A[Original TU graphs] --> B[WL refinement]
  B --> C[Cached persistent V and C]
  C --> D[Magnitude / lambda / binary]
  D --> E[Train-only DictVectorizer]
  E --> F[Random Forest]
  F --> G[Saved nested evaluation]
```

**Presenter notes**

Binary reuses C/V dictionaries and exact saved folds. Global feature matrices are not cached: DictVectorizer must refit on each inner/outer training split.

## Slide 11 — Experimental Setup

**Core content**

Original MUTAG folds: 10 repeats × 10 stratified outer folds; 5 inner folds per outer train. h=0…6, p={1,2}, trees={25,50,100}; λ method additionally selects its prespecified grid. Binary uses the same 42 h/p/tree candidates per outer split. RF: gini, sqrt feature sampling, bootstrap, unlimited depth, min split 2, leaf 1, no class balancing, seed 42, one thread per forest; 8 outer workers. Outer seeds 0…9; inner seeds 1000+10r+f. Select by mean inner accuracy; ties: lower h, p, trees, λ. Accuracy and macro-F1 (zero_division=0). Mean ± population SD over ten fold-averaged repeat means.

**Suggested visualization**

Nested train/validation/test diagram with untouched outer test folds.

**Presenter notes**

Inner selection prevents tuning on outer test scores. Diagnostic outer curve maxima are not final model selections. Maximal 100-tree fits provide independently checked deterministic 25/50-tree prefixes.

## Slide 12 — MUTAG Experimental Results

**Core content**

| Method | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| WL | 88.19 ± 1.23 | 86.70 ± 1.46 |
| P-WL | 86.17 ± 1.20 | 84.25 ± 1.36 |
| P-WL-C | 87.77 ± 1.37 | 86.28 ± 1.47 |
| λ-selected | 87.77 ± 1.26 | 86.28 ± 1.35 |
| P-WL-C-Binary | 86.06 ± 1.16 | 84.30 ± 1.60 |

Best mean accuracy: WL; best mean macro-F1: WL. Binary versus P-WL-C: accuracy -1.71637 pp, macro-F1 -1.98133 pp. Cycles versus P-WL: +1.60526 pp accuracy, +2.03762 pp macro-F1. Full precision is preserved in comparison CSVs.

**Suggested visualization**

`results/binary-cycle/figures/MUTAG_accuracy_comparison.png` and `MUTAG_macro_f1_comparison.png`; distribute table/plots across the slide cleanly.

**Presenter notes**

All rows are completed nested estimates. SD describes repeat variation. Explain very small λ-selected differences using include/exclude selection; do not infer significance.

## Slide 13 — PTC-MR Experimental Results

**Core content**

| Method | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| WL | 62.33 ± 2.25 | 59.54 ± 2.47 |
| P-WL | 62.77 ± 1.03 | 60.04 ± 1.25 |
| P-WL-C | 62.22 ± 1.23 | 59.21 ± 1.51 |
| λ-selected | 62.49 ± 1.55 | 59.62 ± 1.87 |
| P-WL-C-Binary | Not evaluated (out of scope) | Not evaluated (out of scope) |

P-WL has the highest mean accuracy and macro-F1 here. Cycles versus P-WL: −0.55210 pp accuracy, −0.82427 pp macro-F1. Unlike MUTAG, adding cycles reduced the selected-method averages. Binary: **NOT RUN — OUT OF SCOPE**.

**Suggested visualization**

`results/full/PTC_MR_ablation.png`, with the table showing both metrics.

**Presenter notes**

Use only completed existing PTC-MR evidence. The binary row is an explicit scope statement, not an incomplete job awaiting completion.

## Slide 14 — Impact of Each Modification

**Core content**

**MUTAG**

| Change | Accuracy Δ (pp) | Macro-F1 Δ (pp) |
|---|---:|---:|
| WL → P-WL | -2.02632 | -2.45825 |
| P-WL → P-WL-C | +1.60526 | +2.03762 |
| P-WL-C → λ-selected | -0.00292 | -0.00751 |
| P-WL-C → P-WL-C-Binary | -1.71637 | -1.98133 |

WL→P-WL: weight labels by component persistence (not supported as an accuracy improvement here). P-WL→P-WL-C: add cycle information (supported versus P-WL on MUTAG, not PTC-MR). P-WL-C→λ: investigate magnitude influence (positive weights neutral; joint selection adds an include/exclude gate). P-WL-C→Binary: test presence versus magnitude.

**PTC-MR supplementary**

| Change | Accuracy Δ (pp) | Macro-F1 Δ (pp) |
|---|---:|---:|
| WL → P-WL | +0.44034 | +0.49549 |
| P-WL → P-WL-C | -0.55210 | -0.82427 |
| P-WL-C → λ-selected | +0.26471 | +0.40320 |

**Suggested visualization**

A compact change/effect table; use paired binary-versus-magnitude figure as supporting evidence.

**Presenter notes**

Binary encoding changed predictions and reduced mean accuracy. Cycle magnitudes may carry useful information; this is an interpretation, not a causal result. Joint retuning also changed some selected parameters, so the measured difference is between nested method pipelines, not a fixed-model intervention. These are independently selected pipelines and descriptive differences, not a sequence of guaranteed improvements.

## Slide 15 — Discussion and Interpretation

**Core content**

WL remains competitive: richer features do not guarantee useful discriminative information for a finite dataset and classifier. Component persistence reduced MUTAG selected averages, while cycles recovered part of that loss; PTC-MR behaved differently. Positive λ left order-based tree partitions unchanged in measured fits. Binary collapses magnitudes and can change splits.

Binary encoding changed predictions and reduced mean accuracy. Cycle magnitudes may carry useful information; this is an interpretation, not a causal result. Joint retuning also changed some selected parameters, so the measured difference is between nested method pipelines, not a fixed-model intervention.

Predictions changed on 67/100 folds (114/1880 graph-prediction opportunities across repeats). Selected (h,p,trees) changed on 96/100 folds; individual changes: {'h': 83, 'p': 47, 'trees': 67}.

**Suggested visualization**

`results/binary-cycle/figures/MUTAG_paired_fold_differences.png` showing both positive and negative paired changes.

**Presenter notes**

Presence-versus-magnitude interpretation is a hypothesis. Parameter retuning and representation interact. We did not isolate chemistry-level causes or claim statistical significance.

## Slide 16 — Conclusion and Limitations

**Core content**

Studied WL kernels and persistent WL; implemented counts, component weights and cycle contributions; tested custom λ and binary adaptations. Cycles helped versus P-WL on MUTAG and hurt on PTC-MR. Positive λ magnitudes were neutral.

P-WL-C-Binary: accuracy 86.06 ± 1.16%; macro-F1 84.30 ± 1.60%. Versus P-WL-C: accuracy -1.71637 pp; macro-F1 -1.98133 pp.

Small, imbalanced datasets; repeats reuse graphs and folds are dependent. SD is descriptive variability, not a confidence interval. No significance test or causal mechanism is claimed. Binary was requested after observing the earlier study, so this is a follow-up experiment, not an independently preregistered benchmark. Depth 0–6 differs from the paper’s 0–10; inductive vocabularies, fixed modern RF defaults, no scaling, and source-order tie handling are documented adaptations. Edge labels are available but ignored. Binary was evaluated only on MUTAG; no secondary-dataset generalization claim is justified.

Future work: independently held-out benchmarks, justified dependence-aware inference, edge-label-aware representations, and classifiers with different sensitivity to representation. These are proposals, not experiments performed here.

**Suggested visualization**

Three takeaway boxes: established methods, tested modifications, measured evidence.

**Presenter notes**

This directly addresses implement–modify–measure. Preserve negative results. Future PTC-MR binary evaluation is outside this task, not an outstanding deliverable.

## Slide 17 — References

**Core content**

1. Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., and Borgwardt, K. M. (2011). **Weisfeiler-Lehman Graph Kernels.** Journal of Machine Learning Research, 12(77), 2539–2561. [Publisher](https://www.jmlr.org/papers/v12/shervashidze11a.html); local `references/wl-paper.pdf`, `references/wl-paper.txt`.
2. Rieck, B., Bock, C., and Borgwardt, K. (2019). **A Persistent Weisfeiler-Lehman Procedure for Graph Classification.** Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 5448–5458. [Publisher](https://proceedings.mlr.press/v97/rieck19a.html); local `references/paper.pdf`, `references/paper.txt` (Algorithm 1, §2.3.2, Eqs. 9–12).
3. [BorgwardtLab/P-WL authors’ implementation](https://github.com/BorgwardtLab/P-WL). Snapshot identity and file hashes: `references/repository-head.txt`, `references/source_hashes.json`; feature and topology source retained locally. This is implementation evidence, not an extra claim of novelty.
4. [TUDataset catalogue](https://chrsmrrs.github.io/datasets/docs/datasets/) and [bibliography](https://chrsmrrs.github.io/datasets/docs/bibliography/). Dataset origins: Debnath, A. K., Lopez de Compadre, R. L., Debnath, G., Shusterman, A. J., and Hansch, C. (1991), *Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds*, J. Med. Chem. 34(2), 786–797; Helma, C., King, R. D., Kramer, S., and Srinivasan, A. (2001), *The Predictive Toxicology Challenge 2000–2001*, Bioinformatics 17, 107–108. Numerical dataset statistics in this study come from local original TU archives, not rounded online averages.

**Suggested visualization**

Readable citations with publisher links and a small “local evidence” footer.

**Presenter notes**

Paper definitions were checked against downloaded full texts; bibliographic details checked against publisher pages. Numerical experiment claims trace to saved local predictions, not the cited papers’ reported accuracies.

## Evidence appendix

Baseline predictions: `results/full/MUTAG_folds.csv`, `results/full/PTC_MR_folds.csv`; baseline selection evidence: their `*_inner.csv`; original split definitions: `cache/folds/MUTAG-r10.json`, `cache/folds/PTC_MR-r10.json`; labels: `cache/data/{dataset}/{dataset}.zip`, member `{dataset}/{dataset}_graph_labels.txt`. Binary predictions/parameters: `results/binary-cycle/MUTAG_folds.csv`; 4,200 inner candidate means: `MUTAG_inner.csv`; aggregates: `MUTAG_repeat_means.csv`, `MUTAG_summary.csv`; paired evidence: `paired_folds.csv`, `paired_repeat_means.csv`, `comparison.json`. CSV metrics are fractions; differences marked pp are percentage points. Only `setting=selected` rows enter final comparisons.

Dataset statistics: original ZIP graph indicators, graph labels, node labels, adjacency and edge labels; verified statistics also in `results/binary-cycle/dataset_statistics.json`. Tables: `comparison_summary.csv`; full changes: `modification_effects.csv`; binary paired records: `paired_folds.csv`; checks: `verification.json`, `preservation_verified.json`. No PTC-MR binary outputs were generated.
