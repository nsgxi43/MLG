# Beyond Weisfeiler-Lehman — MLG case study

Evaluating Persistent and Cycle-Aware Graph Representations.

This runnable experiment evaluates WL subtree counts, persistence-weighted WL (P-WL), P-WL with cycle magnitudes (P-WL-C), our λ-weighted cycle experiment, and our primary custom representation, **P-WL-C-Binary**. P-WL-C is established in Rieck, Bock and Borgwardt (ICML 2019), not our novel invention; λ weighting and binary encoding are custom adaptations. Source-level decisions and protocol deviations are in [NOTES.md](NOTES.md). This is a reimplementation with a documented evaluation variant, not an exact reproduction of published scores.

Completed: original four methods on MUTAG/PTC-MR (100 folds each), and binary on MUTAG (100 folds). **PTC-MR-Binary: NOT RUN — OUT OF SCOPE; Not evaluated (out of scope).** Original audited artifacts under `results/full/` are preserved. See [RESULTS.md](RESULTS.md) for actual effects, including negative/neutral findings.

## Install and run

Tested with Python 3.9.6 on macOS arm64 using the pinned requirements. Other Python/platform environments have not been validated in this study. Run from this project directory.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest -v test_binary
.venv/bin/python -B run.py --check
.venv/bin/python -B finalize_case_study.py
```

The first command tests the binary transform against cached MUTAG features, split-local matrix dimensions/order, untouched vertex values, original function hashes, cache reuse and checkpoint resumption. The second runs original scientific toy checks, not dataset cross-validation. The third independently verifies saved scores and regenerates only case-study documents/new binary analysis figures; it never trains models or writes into `results/full/`. Do not use legacy `run.py --report-only` for the final five-method case study: its report generator represents the earlier four-method stage.

## Binary experiment: fresh versus resume

The completed new experiment is in `results/binary-cycle/`. A fresh reproduction must use a **new** isolated directory:

```sh
.venv/bin/python -B binary_cycle.py --workers 8 --output results/binary-cycle-fresh
```

To continue an interrupted run with the identical code, dependencies and cached inputs:

```sh
.venv/bin/python -B binary_cycle.py --workers 8 --output results/binary-cycle-fresh --resume
```

The runner refuses an already configured directory without `--resume`. Completed binary runs return without changing their outputs. Interrupted runs reuse completed fold checkpoints, validate input/code/environment hashes, and compute only missing binary folds. Worker count may change without changing the protocol. All new runs are MUTAG-only; there is no PTC-MR binary option. The original h=6 p=1/2 v2 persistence pickles and `cache/folds/MUTAG-r10.json` must be available and valid. Missing required caches cause an explicit error; this runner never silently recomputes persistence or runs a baseline.

For verified analysis of a new completed binary directory: `.venv/bin/python -B finalize_case_study.py --output results/binary-cycle-fresh`. It compares against the preserved `results/full/` baselines. The initial preservation manifest in `results/binary-cycle/preservation_before.json` is audit evidence for this delivered workspace; retain it when reproducing. The generic evaluation pipeline still fits DictVectorizer separately on each inner/outer training split; reusing a globally fitted matrix would introduce vocabulary leakage.

## Original protocol and historical execution

The original corrected full protocol uses ten repeats, h=0..6, p={1,2}, trees={25,50,100}, λ={0,0.25,0.5,1,2}, stratified 10-fold outer and 5-fold inner CV. Eight processes evaluate independent outer folds, with one thread per RF. Binary uses exactly that protocol and the same saved splits, with only its 42 h/p/tree candidates and no λ search. Earlier smoke runs used one repeat, h=0..1, 25 trees and are preliminary.

The measured corrected run used `--reuse-base results/compact-cycle-columns` to reuse unchanged WL/P-WL fits only after checking complete prior input/fold/grid hashes and vertex-block equality. All cycle-enabled fits were recomputed then. Its evaluation times exclude prior reused baseline work. Existing baseline directories, timings, figures and checkpoints must be retained: restarting the legacy entry point into those directories can overwrite metadata/timings even when folds are cached. No baseline rerun is needed for this case study. Historical baseline launch commands are recorded in LOG.md. The original TUDataset loader uses uncleaned archives and verifies labels, targets, counts and edge sets, retaining raw edge order for persistence ties.

## Staged commands used during development (historical, not tasks to execute)

```sh
.venv/bin/python run.py --stage 1 --mode smoke --h-max 6 --output results/stage1
.venv/bin/python run.py --stage 2 --mode smoke --output results/stage2
.venv/bin/python run.py --stage 3 --mode smoke --datasets MUTAG --output results/stage3-mutag
.venv/bin/python run.py --stage 3 --mode smoke --datasets PTC_MR --output results/stage3-ptc
.venv/bin/python run.py --stage 3 --mode full --repeats 1 --h-max 1 --datasets MUTAG PTC_MR --output results/preflight
```

Development-stage timings and scores are preliminary, not final comparisons. See LOG.md for the actual commands and executed checks. Later implementation improvements can change runtime without changing the deterministic scores.

## Method and evaluation

WL encodes the previous node label and the sorted multiset of neighbor labels, then concatenates label counts from 0 through h. P-WL assigns edge weights from label inequality plus neighbor-count Minkowski distance plus 1, inserts edges into an initially vertex-only graph, and sums component persistence raised to p by node label. P-WL-C also sums cycle-birth weights raised to p at both endpoints of each cycle-closing edge. Both blocks use the complete training node-label alphabet, including zero cycle coordinates, so positive-weight matrices have exactly twice P-WL's column count. Existing triangles remain cycles: there is no clique expansion.

Each graph's SHA256 WL signatures and persistence dictionaries are independent of other graphs and cached. Every inner/outer training split fits its own DictVectorizer vocabulary. Validation/test-only features are ignored, never added to a trained forest. Scalers are unnecessary for forests; no whole-dataset normalization is used. The authors' repository processes global label vocabularies and scales outer training before inner splitting; we deliberately use inductive vocabulary construction and avoid that leakage.

Our extension multiplies the separate cycle block by `l` (lambda in figures and text). At `l=0`, cycle columns are removed entirely, giving exactly P-WL, including the feature count used in RF subsampling. Leaving zero columns would not give the same classifier. All nonzero weights are fitted independently; only exactly duplicate zero/one representations share model fits. Random forests use split order rather than metric distances, so positive lambda often changes no predictions. The extension can select whether to include cycles, but positive magnitude is unlikely to provide continuous control.

**P-WL-C-Binary** reuses the exact P-WL-C coordinates and replaces each completed cycle coordinate `c` with `1 if c != 0 else 0`. It does not binarize vertices or individual edges. The unchanged vertex block and transformed cycle block retain all keys, ordering, dimensions and zeros. Unlike positive rescaling, binary presence collapses magnitude distinctions and can change tree partitions. Whether that helps is measured, not assumed; final observed scores and paired changes are in RESULTS.md.

For each outer training set, inner accuracy jointly selects h, p, tree count and lambda as applicable. Ties prefer smaller h, then p, then tree count, then lambda. All methods share the stored outer/inner indices. The diagnostic depth curves fix h and tune remaining settings; lambda curves fix lambda and tune h, p and trees. Never report the best outer curve point as the final estimator. The final ablation uses only `setting=selected` rows.

Accuracy and macro-F1 use fractions in CSVs and percentages in tables/figures. First average ten folds within each repeat, then mean and population SD of the ten repeat means. SD is unavailable for one repeat. Error bars show repeat variability, not confidence intervals. No fold-level significance test is used.

## Outputs

Within each results directory:

* `{dataset}_folds.csv`: method, setting, repeat, outer fold, selected parameters, inner accuracy, outer accuracy/macro-F1, test indices and predictions. `selected` rows are the unbiased nested estimates; `h=...` / `l=...` rows are prespecified diagnostic curves.
* `{dataset}_inner.csv`: every joint candidate's mean inner validation accuracy, for selection auditing.
* `{dataset}_repeat_means.csv`, `{dataset}_summary.csv`, `summary.csv`, `summary.md`: repeat-level and final aggregates.
* `{dataset}_depth.png`, `_ablation.png`, `_lambda.png`, `_runtime.png`: labelled 300 dpi figures with colorblind-safe colors.
* `{dataset}_data.json`: measured sizes, classes, available attributes, ignored edge labels, source URL and archive hash.
* `{dataset}_topology_checks.json` / `_zero_check.json`: actual GUDHI and lambda-zero check counts/results.
* `{dataset}_evaluation.json`: evaluation wall time (inner searches and outer fits), separate from feature construction.
* `feature_runtime.csv` and `{dataset}_runtime_trials.csv`: full-dataset feature construction, cumulative through h, measured in three fresh trials; SD error bars measure timing variability. P-WL and P-WL-C share the persistence pass, so their runtime is explicitly labelled shared. This includes WL label work and persistence dictionaries, excluding disk caching, data download, vocabulary transformation and classifier evaluation.
* `config.json`, `manifest.json`, `checkpoints/`: actual settings, package versions, whole-command wall time, resumable completed folds.

Feature timing is measured afresh in full-stage runs even when data/feature caches exist. Full forests use the same seed, gini criterion, sqrt feature subsampling, bootstrap, unlimited depth, unbalanced class weights and default leaf/split minima. Smaller forest candidates use exact prefixes of a maximal forest; the scientific checks verify prefix predictions against independent smaller fits.

### Binary and case-study outputs

`results/binary-cycle/` contains:

* `MUTAG_folds.csv`: 100 selected outer predictions/parameter rows, in the original evaluator schema.
* `MUTAG_inner.csv`: 4,200 candidate mean inner accuracies (42 per outer fold).
* `MUTAG_repeat_means.csv`, `MUTAG_summary.csv`, `MUTAG_summary.md`: binary-only aggregates.
* `checkpoints/MUTAG-r*-f*.json`, `config.json`, `manifest.json`, `run.log`: resumable run evidence and provenance.
* `comparison_summary.csv`, `modification_effects.csv`: independently verified final comparisons/full-precision method differences.
* `paired_folds.csv`, `paired_repeat_means.csv`, `comparison.json`: identical-split accuracy/macro-F1 differences, win/loss/tie counts, selected parameters and changed predictions.
* `dataset_statistics.json`, `lambda_verification.json`, `verification.json`: raw-archive statistics, retained λ audit and final consistency checks.
* `preservation_before.json`, `preservation_verified.json`: hashes proving 712 original result/cache files and existing method functions were preserved.
* `validation.log`, `scientific_checks.log`: test evidence. The validation log retains the corrected mock-checkpoint fixture failure as well as the subsequent passing run.
* `figures/MUTAG_accuracy_comparison.png`, `MUTAG_macro_f1_comparison.png`, `MUTAG_cycle_binary_comparison.png`, `MUTAG_paired_fold_differences.png`: four new 300 dpi figures. Existing baseline figures are untouched.

No PTC-MR binary predictions, checkpoints or figures exist or are required. Binary timing excludes already-cached persistence construction; no runtime superiority is claimed.

## Files and references

`data.py` loads through PyG TUDataset, converts to NetworkX and restores original TU edge order for persistence ties; `wl.py` caches graph-local refinement; `pwl.py` computes graph persistence and preserves distinct blocks; `evaluate.py` runs nested RF selection; `plots.py` creates outputs; `run.py` is the single entry point; `checks.py` validates scientific properties against GUDHI and original source. `report.py` generates final measured deliverables through run.py. `references/` preserves inspected authors' code and papers for the audit, including the MIT source license.

`binary_cycle.py` performs the isolated cache-only follow-up through the existing evaluator; `test_binary.py` tests transformation/preservation/resume behavior; `finalize_case_study.py` verifies predictions and produces final case-study documents/figures. The original `report.py` remains a legacy four-method report generator. Read RESULTS.md for final interpretation, DEMO.md for the 3–5 minute walkthrough, SLIDES.md for the complete single 17-slide outline, and LOG.md for stage evidence. `results/full/RESULTS.md` remains the immutable earlier baseline report; root RESULTS.md and `results/binary-cycle/RESULTS.md` contain the final case study.

* Rieck, Bock and Borgwardt (2019), [A Persistent Weisfeiler–Lehman Procedure for Graph Classification](https://proceedings.mlr.press/v97/rieck19a.html).
* [Authors' source repository](https://github.com/BorgwardtLab/P-WL), especially features.py, topology.py and grid_search_pwl.py. Its default command explicitly does not reproduce the paper's reported accuracy.
* Shervashidze et al. (2011), [Weisfeiler–Lehman Graph Kernels](https://www.jmlr.org/papers/v12/shervashidze11a.html).
* [TU graph datasets](https://chrsmrrs.github.io/datasets/). MUTAG/PTC_MR archive hashes are recorded in the result metadata.
