"""Generate presentation deliverables only from completed measured results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from data import load_data
from evaluate import summarize
from plots import table


def verify(out):
    out = Path(out)
    config = json.loads((out / 'config.json').read_text())
    assert json.loads((out / 'manifest.json').read_text())['complete']
    summary = pd.read_csv(out / 'summary.csv')
    checks = []
    for name in config['datasets']:
        folds = pd.read_csv(out / f'{name}_folds.csv')
        inner = pd.read_csv(out / f'{name}_inner.csv')
        saved = json.loads(Path(f"cache/folds/{name}-r{config['repeats']}.json").read_text())['folds']
        mapping = {(s['repeat'],s['fold']):s for s in saved}
        _, y, metadata = load_data(name)
        original = json.loads((out / f'{name}_data.json').read_text())
        for field in ['graphs','classes','class_counts','nodes','edges','archive_sha256']:
            assert original[field] == metadata[field]
        (out / f'{name}_tudataset_validation.json').write_text(json.dumps(metadata, indent=2))
        expected_settings = 4 + 3*(config['h_max']+1) + len(config['l'])
        assert len(folds) == expected_settings*config['repeats']*10
        assert not folds.duplicated(['method','setting','repeat','fold']).any()
        assert folds.groupby(['method','setting','repeat']).size().eq(10).all()
        for r in folds.itertuples():
            split = mapping[r.repeat,r.fold]
            test = json.loads(r.test_indices)
            pred = json.loads(r.predictions)
            assert test == split['test'] and len(pred)==len(test)
            assert np.isclose(r.accuracy,accuracy_score(y[test],pred))
            assert np.isclose(r.macro_f1,f1_score(y[test],pred,average='macro',zero_division=0))
            grid = inner[(inner.repeat==r.repeat)&(inner.fold==r.fold)&(inner.method==r.method)]
            if r.setting.startswith('h='):
                grid=grid[grid.h==int(r.setting[2:])]
            elif r.setting.startswith('l='):
                grid=grid[grid.l==float(r.setting[2:])]
            assert np.isclose(r.inner_accuracy,grid.inner_accuracy.max())
            candidate=grid[(grid.h==r.h)&(grid.p==r.p)&(grid.l==r.l)&(grid.trees==r.trees)]
            assert len(candidate)==1 and np.isclose(candidate.iloc[0].inner_accuracy,r.inner_accuracy)
        for repeat in range(config['repeats']):
            rows=folds[(folds.repeat==repeat)&(folds.method=='WL')&(folds.setting=='selected')]
            tested=[i for r in rows.test_indices for i in json.loads(r)]
            assert sorted(tested)==list(range(len(y)))
        _, actual = summarize(folds)
        target=summary[summary.dataset==name].sort_values(['method','setting'])
        actual=actual.sort_values(['method','setting'])
        for field in ['accuracy_mean','accuracy_std','macro_f1_mean','macro_f1_std']:
            assert np.allclose(actual[field],target[field],equal_nan=True)
        a=folds[(folds.method=='P-WL')&(folds.setting=='selected')]
        b=folds[(folds.method=='Extension')&(folds.setting=='l=0')]
        assert a.predictions.tolist()==b.predictions.tolist()
        for figure in ['depth','ablation','lambda','runtime']:
            with Image.open(out/f'{name}_{figure}.png') as im:
                assert im.width>=1800 and abs(im.info['dpi'][0]-300)<1
        trials=pd.read_csv(out/f'{name}_runtime_trials.csv')
        assert trials.trial.nunique()==3 and trials.seconds.gt(0).all()
        checks.append(dict(dataset=name, scored_rows=len(folds), inner_candidates=len(inner),
                           tudataset_identity=True, split_coverage=True, selected_by_inner=True, metrics_recomputed=True,
                           repeat_aggregation=True, zero_equal=True, figures_300dpi=True))
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    return checks


def write_reports(out=Path('results/full')):
    out=Path(out)
    config=json.loads((out/'config.json').read_text())
    assert config['repeats']>=2, 'Final presentation requires repeated measured results'
    checks=verify(out)
    stats=pd.read_csv(out/'summary.csv')
    table(stats,out/'summary.md')
    comparison=[]
    lambda_lines=[]
    demo_findings=[]
    lambda_demo=[]
    timing=[]
    for name in config['datasets']:
        rows=stats[(stats.dataset==name)&(stats.setting=='selected')].set_index('method')
        delta=100*(rows.loc['P-WL-C','accuracy_mean']-rows.loc['P-WL','accuracy_mean'])
        fdelta=100*(rows.loc['P-WL-C','macro_f1_mean']-rows.loc['P-WL','macro_f1_mean'])
        verb='increased' if delta>0 else 'decreased' if delta<0 else 'did not change'
        demo_findings.append(f"{name}: P-WL scored {100*rows.loc['P-WL','accuracy_mean']:.2f}% and P-WL-C "
                             f"{100*rows.loc['P-WL-C','accuracy_mean']:.2f}%; cycles {verb} mean accuracy "
                             f"by {abs(delta):.2f} percentage points.")
        comparison.append(f"On **{name}**, adding cycles {verb} mean accuracy by **{abs(delta):.2f} percentage points** "
                          f"relative to P-WL; macro-F1 changed by {fdelta:+.2f} points. "
                          f"WL scored {100*rows.loc['WL','accuracy_mean']:.2f}%, P-WL {100*rows.loc['P-WL','accuracy_mean']:.2f}%, "
                          f"P-WL-C {100*rows.loc['P-WL-C','accuracy_mean']:.2f}%, and the inner-selected extension "
                          f"{100*rows.loc['Extension','accuracy_mean']:.2f}%. These are descriptive comparisons, without a significance claim.")
        folds=pd.read_csv(out/f'{name}_folds.csv')
        base=folds[(folds.method=='P-WL-C')&(folds.setting=='selected')].sort_values(['repeat','fold'])
        mismatches={}
        for l in config['l']:
            if l==0:
                continue
            test=folds[(folds.method=='Extension')&(folds.setting==f'l={l:g}')].sort_values(['repeat','fold'])
            mismatches[l]=sum(a!=b for a,b in zip(test.predictions,base.predictions))
        selected=folds[(folds.method=='Extension')&(folds.setting=='selected')]
        counts=', '.join(f'lambda={l:g}: {n} folds' for l,n in selected.l.value_counts().sort_index().items())
        if sum(mismatches.values())==0:
            effect=f'All positive weights had exactly identical predictions to P-WL-C on all {len(base)} outer folds.'
        else:
            effect=f'Positive-weight prediction mismatches versus P-WL-C (outer folds): {mismatches}.'
        lambda_lines.append(f'**{name}:** {effect} Zero matched P-WL exactly. Joint inner selection chose {counts}. '
                            'The smallest positive weight wins exact positive-weight ties; that does not establish it as an optimal magnitude.')
        lambda_demo.append(f'{name}: {effect}')
        evaluation=json.loads((out/f'{name}_evaluation.json').read_text())
        runtime=pd.read_csv(out/'feature_runtime.csv')
        maxh=runtime[(runtime.dataset==name)&(runtime.h==config['h_max'])]
        desc=', '.join(f"{r.method} p={r.p}: {r.seconds:.3f} ± {r.seconds_std:.3f} s" for r in maxh.itertuples() if r.method!='P-WL-C')
        timing.append(f"{name}: evaluation wall time {evaluation['evaluation_seconds']:.1f} s with {evaluation['workers']} workers; "
                      f"fresh cumulative feature construction through h={config['h_max']} (3 trials): {desc}. "
                      f"P-WL and P-WL-C use the same timed persistence pass. {evaluation.get('reused_base_folds',0)} unchanged baseline folds were reused; their earlier fit time is excluded from this invocation.")
        table(stats[stats.dataset==name],out/f'{name}_summary.md')
    summary=(out/'summary.md').read_text()
    results=f'''# Final measured results

{summary}
Accuracy and macro-F1 are percentages: mean ± population SD across {config['repeats']} repeat means, each averaging ten outer folds. Error bars show repeat variability, not confidence intervals. The final rows use nested selection; fixed-depth and fixed-lambda curve maxima are not selected final results.

## Do cycle features help?

{chr(10).join(comparison)}

## Effect of lambda (our extension)

{chr(10).join(lambda_lines)}

A random forest partitions feature values by order. Multiplication by a positive constant preserves that order, explaining the flat positive-weight comparison when measured predictions coincide. Lambda zero removes cycle columns, changing both available information and the RF feature-sampling pool. Therefore this extension acts mainly as a choice of whether to include the cycle block. The nested extension can score below a baseline because it makes additional choices from noisy inner-fold estimates; its larger search space guarantees neither better outer accuracy nor better generalization.

## Validation and runtime

Both datasets completed {config['repeats']*10} outer folds. The audit recomputed every saved accuracy and macro-F1 from stored predictions and test labels, confirmed fold coverage and inner-only selection, checked repeat aggregation and all eight 300 dpi figures. Lambda-zero predictions and metrics matched P-WL. GUDHI independently verified every graph's H0/H1 diagram at both p values and all depths. The scientific checks compared attribution against the unmodified authors' calculator, including ties and disconnected graphs.

{chr(10).join(timing)}

Feature and evaluation timing are separate. The runtime figures show cumulative label/persistence construction with SD across three fresh runs; they exclude classifier search and disk overhead. The whole-command duration and software versions are in manifest.json.

## Limitations and inconclusive experiments

These are small, imbalanced benchmark datasets (MUTAG 188 graphs; PTC-MR 344), and repetitions reuse the same graphs. Repeat SD is descriptive; the repeats/folds are not independent new datasets. No naive t-test or Wilcoxon test was applied. The positive lambda magnitude experiment is inconclusive as a mechanism for controlling influence under the required random-forest classifier: when predictions are identical it supplies no evidence for preferring one positive magnitude.

The protocol uses h=0..{config['h_max']} rather than the paper's 0..10, but matches its 10×10 outer / inner 5-fold structure, p={{1,2}}, and the dedicated repository scripts' tree grid {{25,50,100}}. It uses saved shared folds, graph-local hashes, train-only vocabulary fitting, no scaling and current explicitly fixed RF defaults. Original inner folds, historical dependency defaults and dataset-specific balancing choices were not recoverable. The tree grid is verified in the dedicated source scripts; the paper itself omits that grid. Our unbalanced class weights are an explicit protocol choice. These choices prevent test/validation leakage but differ from the authors' dataset-wide vocabulary and preprocessing. Edge labels are available but ignored by this node-labelled method. Vertex attribution and cycle-edge choices in tied filtrations depend on TU node and edge order, following the authors' code; they are not claimed to be invariant to arbitrary reindexing.

The authors' [paper/repository](https://github.com/BorgwardtLab/P-WL) reports accuracy: MUTAG WL 87.26±1.42, P-WL 86.10±1.37, P-WL-C 90.51±1.34%; PTC-MR WL 63.12±1.44, P-WL 63.07±1.68, P-WL-C 64.02±0.82%. This table is contextual reference only. Our results are not an exact reproduction of these figures; see NOTES.md for the source audit and deviations. An initial compact cycle vocabulary failed the source-dimensionality audit because zero cycle columns were omitted. That run was interrupted and retained under results/compact-cycle-columns; it is excluded from final cycle comparisons. The corrected representation retains the complete training node-label alphabet in both blocks. Exact input/fold/grid hashes and vertex-block equality allowed unchanged WL/P-WL fits to be reused; every cycle-enabled model was retrained. All corrected checks passed. Preliminary smoke runs and their differing outcomes are retained in results/ and LOG.md.
'''
    Path('RESULTS.md').write_text(results)
    (out/'RESULTS.md').write_text(results)
    key=' '.join(demo_findings)
    demo=f'''# ML Review 3 demo — approximately 4 minutes

## Before presenting

Open RESULTS.md and results/full/summary.md. Have the depth, ablation, lambda and runtime PNGs ready for both datasets. The final measured result to explain is:

{key}

## Script and outputs

**0:00–0:40 — Problem.** “We classify molecular graphs. Standard WL repeatedly refines node labels from their neighborhoods and counts the resulting subtree patterns. It is efficient, but counts alone do not directly describe how components merge or cycles appear. We ask how much persistent cycle features contribute and whether their relative weight changes prediction.” Show the title and research question.

**0:40–1:25 — Paper method.** “The paper turns label differences into edge weights. We insert edges from small weight to large weight, starting with every vertex at zero. An edge either merges components or closes a cycle. P-WL sums component persistence by node label; P-WL-C adds a separate cycle block at both endpoints. We follow the authors' source for surviving-component caps and stable ties. Triangles are never filled.” Show the method slide and the code's separate blocks in pwl.py.

**1:25–2:00 — Our change and checks.** “Our extension multiplies the cycle block by lambda. Zero removes the block exactly. We checked features, predictions and metrics against P-WL, and checked persistence against GUDHI and the authors' calculator. We tune lambda together with depth, persistence exponent and tree count inside each outer training split.” Show the zero-check JSON and the evaluation slide.

**2:00–2:50 — Measured results.** Read the four methods' values from the final table, then explain the measured cycle change for each dataset: {key} Show both ablation figures. “The values are mean and SD across ten repeat means. This is a descriptive comparison, not a significance claim. Smoke-run results can differ, so the table uses only the completed final protocol.”

**2:50–3:30 — Lambda.** “The classifier required by the paper is a random forest. Positive multiplication preserves feature ordering.” Show both lambda plots and state the recorded behavior: {' '.join(lambda_demo)} “Zero matched P-WL exactly. The extension tests whether to include cycles; it does not establish a useful smooth positive-weight optimum. The smallest positive value wins exact ties, which is not evidence for that magnitude.”

**3:30–4:10 — Runtime and limits.** Show runtime and depth plots. “Construction time is measured separately from nested classifier evaluation. We cache graph-local labels and persistence. Our final depth range is zero to six, whereas the paper used zero to ten. We also use inductive training vocabularies and fixed shared folds. The experiment reimplements the method with documented deviations; it does not claim an exact numerical reproduction. Both gains and losses are reported.”

## Live commands

From the project directory, using the installed environment:

```sh
.venv/bin/python run.py --check
.venv/bin/python run.py --mode smoke --datasets MUTAG PTC_MR --output results/demo-smoke
.venv/bin/python run.py --report-only --output results/full
```

The quick scientific checks fit within a short demo. The smoke command runs a real one-repeat experiment; show existing final results while it runs. Do not label smoke scores as final. To regenerate the full experiment use the final command in README.md; it is not intended as a live four-minute computation.
'''
    Path('DEMO.md').write_text(demo)
    slides=f'''# Slide outline

1. **Problem and research question** — Molecular graph classification; WL subtree counts; contribution of persistent cycle features and the effect of lambda. Title: Reproducing and Extending the Persistent Weisfeiler–Lehman Procedure for Graph Classification.
2. **Paper method** — WL refinement → label-distance edge weights → graph filtration → component/cycle persistence. Separate label-indexed vertex and cycle blocks. Cite Rieck et al. (2019), Shervashidze et al. (2011).
3. **Our modification and validation** — [P-WL, lambda × cycles]; lambda grid {config['l']}; zero equality; authors-code attribution checks; independent GUDHI checks. Positive rescaling is usually invisible to RF.
4. **Evaluation** — MUTAG 188 and PTC-MR 344 graphs; fixed shared {config['repeats']}×10 stratified outer / 5-fold inner CV; h=0..{config['h_max']}, p={{1,2}}, trees={{25,50,100}}; all choices inside inner loop; repeat-level mean ± SD. Training-only vocabularies.
5. **Measured results** — Show final summary and both ablation figures. {' '.join(comparison)} Then show lambda figures and explain identical positive-weight predictions only where verified. Never select a winner from the outer curves.
6. **Runtime and limitations** — Show runtime/depth figures; cached construction versus classifier evaluation; small datasets and overlapping repeats; no fold-level significance claim; source ordering sensitivity; differences from paper, especially depth 0..6 and inductive features. Lambda magnitude is not an effective continuous control with this classifier.
'''
    Path('SLIDES.md').write_text(slides)
    print(f'PASS: final consistency audit for {len(checks)} datasets; RESULTS.md, DEMO.md and SLIDES.md generated',flush=True)
