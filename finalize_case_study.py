"""Verify completed saved predictions, then produce the final case-study artifacts.

No model fitting, persistence generation, downloads, or writes to old results.
"""
import argparse
import hashlib
import io
import inspect
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import evaluate
import pwl
import wl
from binary_cycle import METHOD, ROOT, read_inputs
from plots import COLORS, plt

ORDER = ['WL', 'P-WL', 'P-WL-C', 'Extension', METHOD]
LABELS = {'Extension': 'λ-selected', METHOD: 'P-WL-C-Binary'}
PALETTE = {**COLORS, METHOD: '#56B4E9'}


def labels_from_archive(name):
    path = ROOT / f'cache/data/{name}/{name}.zip'
    with zipfile.ZipFile(path) as z:
        def read(suffix):
            return np.loadtxt(io.BytesIO(z.read(f'{name}/{name}_{suffix}.txt')), dtype=int, delimiter=',')
        raw = read('graph_labels'); ids = read('graph_indicator'); nodes = read('node_labels')
        edges = read('A').reshape(-1, 2)
        undirected = {tuple(sorted(edge)) for edge in edges.tolist()}
        elabels = read('edge_labels')
    stats = dict(graphs=len(raw), classes=np.unique(raw).tolist(),
                 class_counts={str(c): int(sum(raw == c)) for c in np.unique(raw)},
                 nodes=len(ids), edges=len(undirected), node_types=len(np.unique(nodes)),
                 edge_types=len(np.unique(elabels)), avg_nodes=len(ids)/len(raw), avg_edges=len(undirected)/len(raw))
    assert all(ids[a-1] == ids[b-1] for a, b in undirected)
    metadata = json.loads((ROOT / f'results/full/{name}_data.json').read_text())
    for key in ['graphs', 'classes', 'class_counts', 'nodes', 'edges']:
        assert stats[key] == metadata[key], key
    assert stats['node_types'] == len(metadata['node_label_values'])
    assert stats['edge_types'] == len(metadata['edge_label_values'])
    return np.unique(raw, return_inverse=True)[1], stats


def audit_saved(name, out, binary=False):
    df = pd.read_csv(out / f'{name}_folds.csv')
    y, stats = labels_from_archive(name)
    saved = json.loads((ROOT / f'cache/folds/{name}-r10.json').read_text())['folds']
    splits = {(s['repeat'], s['fold']): s for s in saved}
    assert not df.duplicated(['method', 'setting', 'repeat', 'fold']).any()
    assert df.groupby(['method', 'setting', 'repeat']).size().eq(10).all()
    assert set(df.repeat) == set(range(10))
    if binary:
        assert set(df.method) == {METHOD} and set(df.setting) == {'selected'} and len(df) == 100
        assert json.loads((out / 'manifest.json').read_text())['complete']
    else:
        assert len(df) == 3000
    inner = pd.read_csv(out / f'{name}_inner.csv')
    assert len(inner) == (4200 if binary else 31500)
    grouped = {key: g for key, g in inner.groupby(['method', 'repeat', 'fold'])}
    for row in df.itertuples():
        split = splits[(row.repeat, row.fold)]
        test, pred = json.loads(row.test_indices), json.loads(row.predictions)
        assert test == split['test'] and len(test) == len(pred)
        assert abs(row.accuracy - accuracy_score(y[test], pred)) < 1e-14
        assert abs(row.macro_f1 - f1_score(y[test], pred, average='macro', zero_division=0)) < 1e-14
        grid = grouped[row.method, row.repeat, row.fold]
        if row.setting.startswith('h='): grid = grid[grid.h == float(row.setting[2:])]
        if row.setting.startswith('l='): grid = grid[grid.l == float(row.setting[2:])]
        best = grid.sort_values(['inner_accuracy', 'h', 'p', 'trees', 'l'], ascending=[False, True, True, True, True]).iloc[0]
        assert tuple(best[['h', 'p', 'trees', 'l']]) == (row.h, row.p, row.trees, row.l)
        assert abs(best.inner_accuracy - row.inner_accuracy) < 1e-14
    for repeat in range(10):
        indices = [i for split in saved if split['repeat'] == repeat for i in split['test']]
        assert sorted(indices) == list(range(len(y)))
    means, summary = evaluate.summarize(df)
    target = pd.read_csv(out / f'{name}_summary.csv').sort_values(['method', 'setting'])
    actual = summary.sort_values(['method', 'setting'])
    for field in ['accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std']:
        assert np.allclose(actual[field], target[field], atol=1e-14, rtol=0)
    if not binary:
        target = pd.read_csv(out / 'summary.csv'); target = target[target.dataset == name].sort_values(['method', 'setting'])
        for field in ['accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std']:
            assert np.allclose(actual[field], target[field], atol=1e-14, rtol=0)
    if binary:
        cp = list((out / 'checkpoints').glob('MUTAG-r*-f*.json'))
        assert len(cp) == 100 and not list((out / 'checkpoints').glob('PTC*'))
        rows = []; candidates = []
        for path in cp:
            payload = json.loads(path.read_text()); assert len(payload['records']) == 1 and len(payload['inner']) == 42
            rows.extend(payload['records']); candidates.extend(payload['inner'])
        saved_rows = pd.DataFrame(rows).sort_values(['repeat', 'fold']).reset_index(drop=True)
        ordered = df.sort_values(['repeat', 'fold']).reset_index(drop=True)
        for field in ordered.columns:
            if ordered[field].dtype.kind in 'fi': assert np.allclose(ordered[field], saved_rows[field])
            else: assert ordered[field].equals(saved_rows[field])
    return df, means, summary[summary.setting == 'selected'], stats


def lambda_audit(df):
    idx = df.set_index(['method', 'setting', 'repeat', 'fold']).sort_index()
    counts = {}
    for l in ['0', '0.25', '0.5', '1', '2']:
        a = idx.loc[('Extension', 'l='+l)].sort_index()
        b = idx.loc[('P-WL' if l == '0' else 'P-WL-C', 'selected')].sort_index()
        assert a.predictions.equals(b.predictions)
        assert np.array_equal(a[['h', 'p', 'trees']].values, b[['h', 'p', 'trees']].values)
        counts[l] = 0
    chosen = df[(df.method == 'Extension') & (df.setting == 'selected')].l.value_counts()
    return dict(prediction_mismatches=counts, selected_lambda_counts={str(k): int(v) for k, v in chosen.items()})


def preservation(out):
    snapshot = json.loads((ROOT / 'results/binary-cycle/preservation_before.json').read_text())
    for name, expected in snapshot['files'].items():
        path = ROOT / name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Protected artifact changed: {name}'
    modules = {'wl': wl, 'pwl': pwl, 'evaluate': evaluate}
    for qualified, expected in snapshot['functions'].items():
        module, name = qualified.split('.')
        assert hashlib.sha256(inspect.getsource(getattr(modules[module], name)).encode()).hexdigest() == expected
    result = dict(protected_files=len(snapshot['files']), all_existing_results_and_caches_unchanged=True,
                  existing_method_functions_unchanged=True)
    (out / 'preservation_verified.json').write_text(json.dumps(result, indent=2))
    return result


def compare(original, binary):
    a = original[(original.method == 'P-WL-C') & (original.setting == 'selected')]
    b = binary[binary.setting == 'selected']
    pairs = a.merge(b, on=['dataset', 'repeat', 'fold'], suffixes=('_original', '_binary'), validate='one_to_one')
    assert len(pairs) == 100 and pairs.test_indices_original.equals(pairs.test_indices_binary)
    for metric in ['accuracy', 'macro_f1']:
        pairs[metric+'_difference_pp'] = 100*(pairs[metric+'_binary'] - pairs[metric+'_original'])
    pairs['parameters_changed'] = (pairs[['h_original', 'p_original', 'trees_original']].values != pairs[['h_binary', 'p_binary', 'trees_binary']].values).any(axis=1)
    pairs['predictions_changed'] = pairs.predictions_original != pairs.predictions_binary
    pairs['changed_predictions'] = [sum(x != y for x, y in zip(json.loads(a), json.loads(b)))
                                     for a, b in zip(pairs.predictions_original, pairs.predictions_binary)]
    info = {}
    for metric in ['accuracy', 'macro_f1']:
        diff = pairs[metric+'_difference_pp']
        repeat = pairs.groupby('repeat')[metric+'_difference_pp'].mean()
        info[metric] = dict(mean_difference_pp=float(repeat.mean()), repeat_difference_sd_pp=float(repeat.std(ddof=0)),
                            improved=int((diff > 1e-12).sum()), worsened=int((diff < -1e-12).sum()), tied=int((diff.abs() <= 1e-12).sum()),
                            min_fold_difference_pp=float(diff.min()), max_fold_difference_pp=float(diff.max()))
    info['parameters_changed_folds'] = int(pairs.parameters_changed.sum())
    info['parameter_change_counts'] = {key: int((pairs[key+'_original'] != pairs[key+'_binary']).sum()) for key in ['h', 'p', 'trees']}
    info['predictions_changed_folds'] = int(pairs.predictions_changed.sum())
    info['changed_predictions'] = int(pairs.changed_predictions.sum())
    info['prediction_opportunities'] = sum(len(json.loads(x)) for x in pairs.test_indices_original)
    return pairs, info


def markdown_table(stats, ptc=False):
    lines = ['| Method | Accuracy (%) | Macro-F1 (%) |', '|---|---:|---:|']
    for method in ORDER:
        if method == METHOD and ptc:
            lines.append('| P-WL-C-Binary | Not evaluated (out of scope) | Not evaluated (out of scope) |'); continue
        row = stats[stats.method == method].iloc[0]
        lines.append(f"| {LABELS.get(method, method)} | {100*row.accuracy_mean:.2f} ± {100*row.accuracy_std:.2f} | {100*row.macro_f1_mean:.2f} ± {100*row.macro_f1_std:.2f} |")
    return '\n'.join(lines)


def impact_table(stats):
    s = stats.set_index('method'); lines = ['| Change | Accuracy Δ (pp) | Macro-F1 Δ (pp) |', '|---|---:|---:|']
    for a, b in [('WL', 'P-WL'), ('P-WL', 'P-WL-C'), ('P-WL-C', 'Extension'), ('P-WL-C', METHOD)]:
        if b not in s.index: continue
        lines.append(f"| {LABELS.get(a,a)} → {LABELS.get(b,b)} | {100*(s.loc[b,'accuracy_mean']-s.loc[a,'accuracy_mean']):+.5f} | {100*(s.loc[b,'macro_f1_mean']-s.loc[a,'macro_f1_mean']):+.5f} |")
    return '\n'.join(lines)


def make_figures(stats, pairs, out):
    dest = out / 'figures'; dest.mkdir(exist_ok=True)
    values = stats.set_index('method').loc[ORDER]
    for metric, label in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(values)); means = 100*values[metric+'_mean']; errors = 100*values[metric+'_std']
        ax.bar(x, means, yerr=errors, color=[PALETTE[m] for m in ORDER], capsize=4)
        for i, value in enumerate(means): ax.text(i, value+errors.iloc[i]+1, f'{value:.2f}', ha='center', fontsize=11)
        ax.set(xticks=x, xticklabels=['WL', 'P-WL', 'P-WL-C', 'λ-selected', 'P-WL-C\nBinary'], ylim=(0, 100),
               ylabel=label+' (%)', title=f'MUTAG: {label.lower()} — nested selection, 10 repeats')
        ax.text(.5, -.18, 'Error bars: population SD of ten repeat means; not confidence intervals.', transform=ax.transAxes, ha='center', fontsize=9)
        fig.tight_layout(); fig.savefig(dest / f'MUTAG_{metric}_comparison.png', dpi=300); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, metric, label in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
        subset = values.loc[['P-WL-C', METHOD]]; x = np.arange(2)
        ax.bar(x, 100*subset[metric+'_mean'], yerr=100*subset[metric+'_std'], color=[PALETTE[m] for m in subset.index], capsize=4)
        ax.set(xticks=x, xticklabels=['P-WL-C', 'P-WL-C\nBinary'], ylim=(0, 100), ylabel=label+' (%)')
        delta = 100*(subset.loc[METHOD,metric+'_mean']-subset.loc['P-WL-C',metric+'_mean'])
        ax.set_title(f'Binary − magnitude: {delta:+.3f} pp')
    fig.suptitle('MUTAG: cycle magnitude versus presence (mean ± repeat SD)')
    fig.tight_layout(); fig.savefig(dest / 'MUTAG_cycle_binary_comparison.png', dpi=300); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ordered = pairs.sort_values(['repeat', 'fold'])
    for ax, metric, label in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
        diff = ordered[metric+'_difference_pp'].to_numpy()
        ax.scatter(np.arange(1, 101), diff, c=np.where(diff>1e-12, '#009E73', np.where(diff < -1e-12, '#D55E00', '#777777')), s=20)
        ax.axhline(0, color='black', linewidth=.8); ax.set(ylabel=label+' difference (pp)')
        for x in range(10, 100, 10): ax.axvline(x+.5, color='#dddddd', linewidth=.7)
    axes[-1].set_xlabel('Paired outer-fold record (repeat 1–10, ten folds each)')
    fig.suptitle('MUTAG: binary − magnitude on identical splits\nDescriptive paired differences; folds are dependent')
    fig.tight_layout(); fig.savefig(dest / 'MUTAG_paired_fold_differences.png', dpi=300); plt.close(fig)


def write_docs(mutag, ptc, dataset_stats, comparison, lambdas, out):
    M = mutag.set_index('method'); B = M.loc[METHOD]; C = M.loc['P-WL-C']
    delta = comparison['accuracy']['mean_difference_pp']; fdelta = comparison['macro_f1']['mean_difference_pp']
    table_m, table_p = markdown_table(mutag), markdown_table(ptc, ptc=True)
    best = M.accuracy_mean.idxmax(); best_f1 = M.macro_f1_mean.idxmax()
    observed = f"P-WL-C-Binary: accuracy {100*B.accuracy_mean:.2f} ± {100*B.accuracy_std:.2f}%; macro-F1 {100*B.macro_f1_mean:.2f} ± {100*B.macro_f1_std:.2f}%. Versus P-WL-C: accuracy {delta:+.5f} pp; macro-F1 {fdelta:+.5f} pp."
    counts = '\n'.join(f"- {metric}: {comparison[metric]['improved']} folds improved, {comparison[metric]['worsened']} worsened, {comparison[metric]['tied']} tied." for metric in ['accuracy', 'macro_f1'])
    changed = f"Predictions changed on {comparison['predictions_changed_folds']}/100 folds ({comparison['changed_predictions']}/{comparison['prediction_opportunities']} graph-prediction opportunities across repeats). Selected (h,p,trees) changed on {comparison['parameters_changed_folds']}/100 folds; individual changes: {comparison['parameter_change_counts']}."
    if comparison['predictions_changed_folds'] == 0:
        interpretation = 'The representation changed, but it had no observed predictive effect under this evaluation.'
    elif delta < 0:
        interpretation = 'Binary encoding changed predictions and reduced mean accuracy. Cycle magnitudes may carry useful information; this is an interpretation, not a causal result. Joint retuning also changed some selected parameters, so the measured difference is between nested method pipelines, not a fixed-model intervention.'
    elif delta > 0:
        interpretation = 'Binary encoding changed predictions and increased mean accuracy in this MUTAG evaluation. Presence indicators may reduce sensitivity to magnitudes; this is a hypothesis, not a proven mechanism or a universal improvement. The comparison includes independently nested parameter selection.'
    else:
        interpretation = 'Binary encoding changed predictions but left mean accuracy unchanged. Equal aggregate accuracy does not imply identical models or identical errors.'
    lambda_text = 'On both datasets, λ=0 exactly matched P-WL; λ∈{0.25,0.5,1,2} matched P-WL-C predictions and selected h/p/tree parameters on all 100 outer folds. Positive rescaling preserves ordering and therefore available axis-aligned split partitions in exact arithmetic. This explains the measured neutrality, but is not a guarantee across models, numerical implementations, or protocols. Joint λ selection can choose zero, so the λ-selected estimator need not equal P-WL-C.'
    limits = 'Small, imbalanced datasets; repeats reuse graphs and folds are dependent. SD is descriptive variability, not a confidence interval. No significance test or causal mechanism is claimed. Binary was requested after observing the earlier study, so this is a follow-up experiment, not an independently preregistered benchmark. Depth 0–6 differs from the paper’s 0–10; inductive vocabularies, fixed modern RF defaults, no scaling, and source-order tie handling are documented adaptations. Edge labels are available but ignored. Binary was evaluated only on MUTAG; no secondary-dataset generalization claim is justified.'
    references = '''1. Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., and Borgwardt, K. M. (2011). **Weisfeiler-Lehman Graph Kernels.** Journal of Machine Learning Research, 12(77), 2539–2561. [Publisher](https://www.jmlr.org/papers/v12/shervashidze11a.html); local `references/wl-paper.pdf`, `references/wl-paper.txt`.
2. Rieck, B., Bock, C., and Borgwardt, K. (2019). **A Persistent Weisfeiler-Lehman Procedure for Graph Classification.** Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 5448–5458. [Publisher](https://proceedings.mlr.press/v97/rieck19a.html); local `references/paper.pdf`, `references/paper.txt` (Algorithm 1, §2.3.2, Eqs. 9–12).
3. [BorgwardtLab/P-WL authors’ implementation](https://github.com/BorgwardtLab/P-WL). Snapshot identity and file hashes: `references/repository-head.txt`, `references/source_hashes.json`; feature and topology source retained locally. This is implementation evidence, not an extra claim of novelty.
4. [TUDataset catalogue](https://chrsmrrs.github.io/datasets/docs/datasets/) and [bibliography](https://chrsmrrs.github.io/datasets/docs/bibliography/). Dataset origins: Debnath, A. K., Lopez de Compadre, R. L., Debnath, G., Shusterman, A. J., and Hansch, C. (1991), *Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds*, J. Med. Chem. 34(2), 786–797; Helma, C., King, R. D., Kramer, S., and Srinivasan, A. (2001), *The Predictive Toxicology Challenge 2000–2001*, Bioinformatics 17, 107–108. Numerical dataset statistics in this study come from local original TU archives, not rounded online averages.'''
    sources = 'Baseline predictions: `results/full/MUTAG_folds.csv`, `results/full/PTC_MR_folds.csv`; baseline selection evidence: their `*_inner.csv`; original split definitions: `cache/folds/MUTAG-r10.json`, `cache/folds/PTC_MR-r10.json`; labels: `cache/data/{dataset}/{dataset}.zip`, member `{dataset}/{dataset}_graph_labels.txt`. Binary predictions/parameters: `results/binary-cycle/MUTAG_folds.csv`; 4,200 inner candidate means: `MUTAG_inner.csv`; aggregates: `MUTAG_repeat_means.csv`, `MUTAG_summary.csv`; paired evidence: `paired_folds.csv`, `paired_repeat_means.csv`, `comparison.json`. CSV metrics are fractions; differences marked pp are percentage points. Only `setting=selected` rows enter final comparisons.'
    results = f'''# Final MLG case-study results

## Completed work and scope

Corrected WL, P-WL, P-WL-C and λ experiments: MUTAG and PTC-MR, 100 outer folds each. New P-WL-C-Binary: MUTAG, 100/100 folds, 0 missing. **PTC-MR-Binary: NOT RUN — OUT OF SCOPE; Not evaluated (out of scope).** No baseline experiment was rerun. All 712 pre-existing result/cache files passed byte-for-byte preservation checks; original method functions are unchanged. Evidence: `results/binary-cycle/preservation_verified.json`, `verification.json`, `manifest.json`.

## MUTAG — primary dataset

{table_m}

Mean ± population SD across ten repeat means, each averaging ten outer folds. Best mean accuracy: **{best}**; best mean macro-F1: **{best_f1}**. Values rounded to two decimals can hide small differences. Full-precision changes are below.

{impact_table(mutag)}

P-WL-C added +1.60526 pp accuracy and +2.03762 pp macro-F1 over P-WL, despite remaining below WL. The saved accuracy difference rounds to +1.61 pp. Its observed effect differs between datasets.

## Binary versus magnitude — paired evidence

{observed}

{counts}

{changed}

Paired fold accuracy differences range {comparison['accuracy']['min_fold_difference_pp']:+.5f} to {comparison['accuracy']['max_fold_difference_pp']:+.5f} pp; macro-F1 differences range {comparison['macro_f1']['min_fold_difference_pp']:+.5f} to {comparison['macro_f1']['max_fold_difference_pp']:+.5f} pp. Across ten paired repeat means, difference SD is {comparison['accuracy']['repeat_difference_sd_pp']:.5f} pp for accuracy and {comparison['macro_f1']['repeat_difference_sd_pp']:.5f} pp for macro-F1. These are descriptive paired summaries, without significance claims.

{interpretation}

## PTC-MR — supplementary completed evidence

{table_p}

{impact_table(ptc)}

Cycles reduced mean accuracy by 0.55210 pp and macro-F1 by 0.82427 pp versus P-WL. P-WL has the highest mean accuracy and macro-F1 here. Method behavior differs across datasets; binary has not been tested here.

## Retained λ experiment

{lambda_text}

Inner λ selection: MUTAG {lambdas['MUTAG']['selected_lambda_counts']}; PTC-MR {lambdas['PTC_MR']['selected_lambda_counts']}. The smallest positive weight wins exact ties; it is not an empirically superior magnitude.

## Verification, sources, and figures

{sources}

Independent verification recomputed every saved baseline and binary score from labels and predictions, checked repeat aggregation, selection against the saved inner argmax with deterministic ties, checkpoint/CSV agreement, original folds, and baseline preservation. No persistence was regenerated: binary read the corrected v2 h=6 p=1/2 feature pickles. Existing GUDHI checks remain preserved; the new transformation needs no new filtration.

New 300 dpi figures: `results/binary-cycle/figures/MUTAG_accuracy_comparison.png`, `MUTAG_macro_f1_comparison.png`, `MUTAG_cycle_binary_comparison.png`, `MUTAG_paired_fold_differences.png`. Original `results/full/` figures are unchanged. Timing is in the binary evaluation JSON and manifest and excludes pre-existing feature construction. No speed improvement is claimed.

## Archived and preliminary work

`results/compact-cycle-columns/` is the interrupted compact-column attempt: MUTAG 100 folds, PTC-MR 39/100 (61 missing). Its cycle/extension results, including MUTAG ~88.20%, are superseded and excluded. WL/P-WL were reused only after original hash/vertex-block checks. Stage/smoke/preflight/resume-check outputs are development evidence, not final score sources. `results/MUTAG_component_audit.json` is an unversioned historical audit; the corrected and binary verification files are the relevant evidence.

## Limitations

{limits}

## Literature and attribution

P-WL-C is already a literature-based cycle enhancement in Rieck et al. (2019), §2.3.2; it is a core implementation in our study, not our novel invention. λ weighting and binary cycle presence are our custom adaptations. Implementing, modifying, and measuring includes negative and neutral findings; higher accuracy is not guaranteed.

{references}
'''
    (ROOT / 'RESULTS.md').write_text(results)
    (out / 'RESULTS.md').write_text(results)
    demo = f'''# MLG walkthrough — 4 minutes 30 seconds

Use the 17-slide outline in SLIDES.md as supporting material; the short walkthrough emphasizes slides 1, 5–9, 11–16. Open existing files before presenting; no experiment needs to run live.

**0:00–0:30 — Problem and research.** “We classify molecular graphs from atom labels and connectivity. We studied Shervashidze et al.’s WL kernels and Rieck et al.’s persistent WL procedure. Our question is whether component information, cycle magnitudes, or cycle presence changes predictive performance.” MUTAG is primary; completed PTC-MR results are supplementary.

**0:30–1:05 — WL and P-WL.** “WL refines each node label with the multiset of neighbor labels and concatenates counts across depths. Their dot product defines a kernel; our classifier instead receives explicit vectors in a Random Forest. P-WL uses label distances to order existing edges. Merges produce component persistence features.” Show `wl.py` and the pipeline on slide 10.

**1:05–1:40 — Cycles.** “Edges whose endpoints are already connected close cycles. P-WL-C sums their birth-weight powers by both endpoint labels in a separate block. Triangles remain cycles: we do not fill them. This is a paper-derived enhancement, and it improved MUTAG accuracy by 1.61 points over P-WL, even though WL remained higher. On PTC-MR, cycles reduced accuracy by 0.55 points.” Show the existing ablation evidence.

**1:40–2:10 — λ attempt and binary change.** “Our first adaptation scaled the cycle block. Zero exactly reproduces P-WL. All tested positive weights gave identical predictions and selected parameters to P-WL-C. Trees can be insensitive to positive rescaling because ordering is unchanged. Our primary new modification instead maps each completed cycle coordinate to presence: nonzero becomes one, zero stays zero. We leave every vertex value, key, dimension, and ordering unchanged.” Show `pwl.combine_binary` and slide 9.

**2:10–2:40 — Fair evaluation.** “We reused the original 100 MUTAG outer splits and their five inner folds. Inner accuracy selects depth 0–6, p=1 or 2, and 25/50/100 trees. Vocabularies fit on training data only. Accuracy and macro-F1 are averaged within each repeat, then summarized across ten repeats. Baselines and persistence caches were preserved.” Show verification and slide 11.

**2:40–3:35 — Actual results.** {observed} “The highest mean accuracy is {best}; the highest mean macro-F1 is {best_f1}. Cycles are useful relative to P-WL on MUTAG, while method performance differs on PTC-MR.” Show `results/binary-cycle/figures/MUTAG_accuracy_comparison.png` and `MUTAG_macro_f1_comparison.png`.

**3:35–4:05 — Paired changes.** “Accuracy: {comparison['accuracy']['improved']} improved folds, {comparison['accuracy']['worsened']} worsened, {comparison['accuracy']['tied']} tied. Predictions changed on {comparison['predictions_changed_folds']} folds, and selected parameter triples changed on {comparison['parameters_changed_folds']}. These are descriptive paired results, not significance evidence.” Show `MUTAG_paired_fold_differences.png`.

**4:05–4:30 — Lessons and limits.** “{interpretation} Small datasets and overlapping repeats limit inference. Our project implements, modifies, and measures representations; it does not promise every change improves performance. PTC-MR-Binary was not evaluated: out of scope.”

For validation before the presentation: `.venv/bin/python -B -m unittest -v test_binary`. To regenerate verified case-study reports and new figures without training: `.venv/bin/python -B finalize_case_study.py`. These write only the case-study documents/new binary analysis outputs; they preserve `results/full/`. Keep the baseline `run.py --report-only` command out of this workflow because it regenerates the older four-method documents.
'''
    (ROOT / 'DEMO.md').write_text(demo)
    ds = dataset_stats
    dataset_table = '\n'.join(['| Dataset | Graphs / classes | Class counts (-1,+1) | Nodes / undirected edges | Atom / bond label types | Average nodes / edges |', '|---|---|---|---|---|---|'] +
        [f"| {name} | {s['graphs']} / 2 | {s['class_counts']['-1']}, {s['class_counts']['1']} | {s['nodes']} / {s['edges']} | {s['node_types']} / {s['edge_types']} | {s['avg_nodes']:.5f} / {s['avg_edges']:.5f} |" for name,s in ds.items()])
    slide_rows = [
      ('Title and Case Study Objective',
       '**Beyond Weisfeiler-Lehman: Evaluating Persistent and Cycle-Aware Graph Representations**\n\nMachine Learning on Graphs case study. Implement established WL representations; investigate how modifications to topological cycle features affect graph classification. Completed: WL, P-WL, P-WL-C and λ on both datasets; binary on MUTAG only.',
       'Title with the five representation names; highlight the binary variant as the primary custom representation.',
       'Our goal is implementation, controlled modification, and measurement, not guaranteed improvement.'),
      ('Problem Statement and Motivation',
       'Graph classification assigns a label to an entire graph. Tabular counts can omit connectivity; labeled neighborhood patterns retain more structure. WL offers efficient explicit features and induced graph kernels, but can miss global connectivity and cycles. Research question: do persistence, cycle magnitudes, weighting, and cycle presence change classification?',
       'A uniform six-cycle beside two disjoint triangles: same degree/local WL labels, different component/cycle counts.',
       'This counterexample illustrates a limitation, not a claim that all WL features ignore every cycle. `checks.py` validates the example.'),
      ('Datasets: MUTAG and PTC-MR',
       dataset_table+'\n\nMUTAG: molecular mutagenicity classification, primary binary dataset. PTC-MR: molecular carcinogenicity classification for male rats, supplementary earlier-method evidence. Nodes represent atoms, edges bonds; both have categorical node/edge labels and no continuous node/edge attributes. The implemented methods use node labels and topology, ignoring edge labels. These small established benchmarks allow controlled, affordable nested comparisons involving cyclic molecular structures.',
       'The dataset table plus one schematic atom/bond graph; use drawings only as illustrations, not reconstructed chemistry.',
       'All numeric statistics were recomputed from local original TU archives and checked against `_data.json`; raw classes are -1/+1. PTC-MR-Binary: Not evaluated (out of scope).'),
      ('Research Paper Foundation',
       'Shervashidze et al. (2011): efficient kernels for labeled graph classification via iterative neighborhood-label refinement and concatenated subtree-pattern counts; local refinement can miss global topology. Rieck et al. (2019): enrich WL features using filtrations and persistent connected-component/cycle information. Its Algorithm 1 and §2.3.2 define P-WL and P-WL-C; cycle inclusion is established literature, not our novel invention. Our source-following caps/ties, inductive vocabularies, h≤6, and modern RF defaults are documented choices; λ and binary are custom adaptations.\n\nFull citations: slide 17.',
       'Two paper-title cards with “neighborhood counts” → “persistent components and cycles.”',
       'Paper-inspired reimplementation with deviations, not exact reproduction of published scores. The original source resolves attribution details the paper leaves underspecified.'),
      ('Original Weisfeiler-Lehman Method (WL)',
       '`l₀(v)` is the categorical node label. `lₕ(v) = hash(lₕ₋₁(v), sorted multiset of neighbor labels)`. `φ_WL(G)` concatenates label-count coordinates for iterations 0…h. Example: a center labeled A with B,B neighbors gets a different refined signature from A with B,C neighbors. The induced kernel is `k(G,G′)=〈φ(G),φ(G′)〉`; experiments feed explicit vectors to a Random Forest, without constructing a Gram matrix.',
       'Two labeled three-node neighborhoods and their distinct refined signatures, then a count vector.',
       'Implementation: `wl.py` signature, wl_features, cumulative. SHA256 names keep graph-local caching independent of other graphs.'),
      ('Persistent Weisfeiler-Lehman (P-WL)',
       'At h=0: `w(u,v)=1+1[label(u)≠label(v)]`. At h>0: add the Minkowski p-distance between previous-label neighbor-count vectors. Begin with vertices at zero and insert existing edges in stable increasing weight order. Component merges give deaths; surviving components are capped at the maximum edge weight (edgeless fallback 1). `Vₕ(a)=Σ_{v:lₕ(v)=a} death(v)^p`, p∈{1,2}; concatenate h=0…H. Unlike WL counts, vertex occurrences are persistence-weighted.',
       'A four-node weighted edge-insertion sequence showing a merge and component death.',
       'Implementation: `pwl.edge_weights`, `pwl.persistence`, `pwl.pwl_features`. The same p controls edge distance and contribution powers. Attribution follows source index order, not an arbitrary reindexing-invariance claim.'),
      ('Cycle-Feature Enhancement (P-WL-C)',
       'A graph cycle is a closed route; cycle information complements merge-only H0 features. If an inserted edge connects already-connected vertices, it births H1. `Cₕ(a)=Σ_{e=(u,v) closing a cycle} w(e)^p × (1[lₕ(u)=a]+1[lₕ(v)=a])`. Same-label endpoints contribute twice. `φ_P-WL-C=[V₀…V_H, C₀…C_H]` with depth-namespaced coordinates and complete zero-inclusive cycle alphabet. No triangles are filled.\n\nMUTAG: P-WL 86.17%, P-WL-C 87.77%; exact +1.60526 pp (+1.61 rounded); WL 88.19%. PTC-MR: P-WL 62.77%, P-WL-C 62.22%; −0.55210 pp.',
       'Highlight the final cycle-closing edge and both endpoint-label bins; show `results/full/MUTAG_ablation.png`.',
       'This paper-derived feature enhancement is a core part of our implemented study. It helped versus P-WL on MUTAG and hurt on PTC-MR; remaining below WL does not make it irrelevant.'),
      ('λ-Weighted Cycle-Feature Extension',
       '`φλ=[V, λC]`, λ∈{0,0.25,0.5,1,2}. λ=0 physically removes cycle columns and equals P-WL, including RF feature sampling. Every positive tested value yielded identical predictions and selected h/p/tree parameters to P-WL-C on all 100 folds per dataset. Scaling preserves value order; axis-aligned trees often retain split partitions. The λ-selected model also chooses include/exclude, so its final scores can differ slightly.',
       '`results/full/MUTAG_lambda.png` and `results/full/PTC_MR_lambda.png`.',
       'Keep this legitimate but empirically ineffective magnitude tweak visible. Do not call 0.25 optimal: it wins exact ties. No universal invariance guarantee.'),
      ('Proposed Binary Cycle Representation (P-WL-C-Binary)',
       'Hypothesis: can presence of a cycle pattern be more informative than its magnitude? `Bₕ(a)=1[Cₕ(a)≠0]`; `φ_binary=[V,B]`. Binarize the completed aggregate coordinate, not each edge. V is untouched; keys, ordering, zero-inclusive vocabulary, and dimensions are preserved. Example: `[0,2,8]→[0,1,1]`. Positive values collapse into one category, removing magnitude distinctions; this is a representation change unlike positive λ rescaling. MUTAG only.',
       'Side-by-side magnitude and presence vectors; underline identical V blocks.',
       'Implementation: `pwl.combine_binary`. It reads existing corrected caches through `binary_cycle.read_inputs`; no new persistence or graph preprocessing.'),
      ('Implementation and System Architecture',
       'Dataset → preprocessing → WL refinement → persistent V/C → feature variants → training-only vocabulary → Random Forest → nested evaluation. `data.py`: original TUDataset/NetworkX preprocessing; `wl.py`: label refinement; `pwl.py`: persistence/variants; `evaluate.py`: shared splits, RF, selection, checkpoints; `binary_cycle.py`: cache-only MUTAG runner; `test_binary.py`: transformation/resume regression checks; `finalize_case_study.py`: prediction audit, comparisons, figures, reports. Cached graph-local features are transformed without globally fitting vocabularies.',
       '```mermaid\nflowchart LR\n  A[Original TU graphs] --> B[WL refinement]\n  B --> C[Cached persistent V and C]\n  C --> D[Magnitude / lambda / binary]\n  D --> E[Train-only DictVectorizer]\n  E --> F[Random Forest]\n  F --> G[Saved nested evaluation]\n```',
       'Binary reuses C/V dictionaries and exact saved folds. Global feature matrices are not cached: DictVectorizer must refit on each inner/outer training split.'),
      ('Experimental Setup',
       'Original MUTAG folds: 10 repeats × 10 stratified outer folds; 5 inner folds per outer train. h=0…6, p={1,2}, trees={25,50,100}; λ method additionally selects its prespecified grid. Binary uses the same 42 h/p/tree candidates per outer split. RF: gini, sqrt feature sampling, bootstrap, unlimited depth, min split 2, leaf 1, no class balancing, seed 42, one thread per forest; 8 outer workers. Outer seeds 0…9; inner seeds 1000+10r+f. Select by mean inner accuracy; ties: lower h, p, trees, λ. Accuracy and macro-F1 (zero_division=0). Mean ± population SD over ten fold-averaged repeat means.',
       'Nested train/validation/test diagram with untouched outer test folds.',
       'Inner selection prevents tuning on outer test scores. Diagnostic outer curve maxima are not final model selections. Maximal 100-tree fits provide independently checked deterministic 25/50-tree prefixes.'),
      ('MUTAG Experimental Results',
       table_m+f'\n\nBest mean accuracy: {best}; best mean macro-F1: {best_f1}. Binary versus P-WL-C: accuracy {delta:+.5f} pp, macro-F1 {fdelta:+.5f} pp. Cycles versus P-WL: +1.60526 pp accuracy, +2.03762 pp macro-F1. Full precision is preserved in comparison CSVs.',
       '`results/binary-cycle/figures/MUTAG_accuracy_comparison.png` and `MUTAG_macro_f1_comparison.png`; distribute table/plots across the slide cleanly.',
       'All rows are completed nested estimates. SD describes repeat variation. Explain very small λ-selected differences using include/exclude selection; do not infer significance.'),
      ('PTC-MR Experimental Results',
       table_p+'\n\nP-WL has the highest mean accuracy and macro-F1 here. Cycles versus P-WL: −0.55210 pp accuracy, −0.82427 pp macro-F1. Unlike MUTAG, adding cycles reduced the selected-method averages. Binary: **NOT RUN — OUT OF SCOPE**.',
       '`results/full/PTC_MR_ablation.png`, with the table showing both metrics.',
       'Use only completed existing PTC-MR evidence. The binary row is an explicit scope statement, not an incomplete job awaiting completion.'),
      ('Impact of Each Modification',
       '**MUTAG**\n\n'+impact_table(mutag)+'\n\nWL→P-WL: weight labels by component persistence (not supported as an accuracy improvement here). P-WL→P-WL-C: add cycle information (supported versus P-WL on MUTAG, not PTC-MR). P-WL-C→λ: investigate magnitude influence (positive weights neutral; joint selection adds an include/exclude gate). P-WL-C→Binary: test presence versus magnitude.\n\n**PTC-MR supplementary**\n\n'+impact_table(ptc),
       'A compact change/effect table; use paired binary-versus-magnitude figure as supporting evidence.',
       interpretation+' These are independently selected pipelines and descriptive differences, not a sequence of guaranteed improvements.'),
      ('Discussion and Interpretation',
       'WL remains competitive: richer features do not guarantee useful discriminative information for a finite dataset and classifier. Component persistence reduced MUTAG selected averages, while cycles recovered part of that loss; PTC-MR behaved differently. Positive λ left order-based tree partitions unchanged in measured fits. Binary collapses magnitudes and can change splits.\n\n'+interpretation+'\n\n'+changed,
       '`results/binary-cycle/figures/MUTAG_paired_fold_differences.png` showing both positive and negative paired changes.',
       'Presence-versus-magnitude interpretation is a hypothesis. Parameter retuning and representation interact. We did not isolate chemistry-level causes or claim statistical significance.'),
      ('Conclusion and Limitations',
       'Studied WL kernels and persistent WL; implemented counts, component weights and cycle contributions; tested custom λ and binary adaptations. Cycles helped versus P-WL on MUTAG and hurt on PTC-MR. Positive λ magnitudes were neutral.\n\n'+observed+'\n\n'+limits+'\n\nFuture work: independently held-out benchmarks, justified dependence-aware inference, edge-label-aware representations, and classifiers with different sensitivity to representation. These are proposals, not experiments performed here.',
       'Three takeaway boxes: established methods, tested modifications, measured evidence.',
       'This directly addresses implement–modify–measure. Preserve negative results. Future PTC-MR binary evaluation is outside this task, not an outstanding deliverable.'),
      ('References', references,
       'Readable citations with publisher links and a small “local evidence” footer.',
       'Paper definitions were checked against downloaded full texts; bibliographic details checked against publisher pages. Numerical experiment claims trace to saved local predictions, not the cited papers’ reported accuracies.')]
    slides = ['# Comprehensive MLG case-study presentation — 17 slides', '',
              'This is the single faculty-ready outline. Tables are generated from audited saved predictions; see RESULTS.md and the traceability appendix below.', '']
    for i, (title, content, visual, notes) in enumerate(slide_rows, 1):
        slides.extend([f'## Slide {i} — {title}', '', '**Core content**', '', content, '', '**Suggested visualization**', '', visual, '', '**Presenter notes**', '', notes, ''])
    slides.extend(['## Evidence appendix', '', sources, '',
        'Dataset statistics: original ZIP graph indicators, graph labels, node labels, adjacency and edge labels; verified statistics also in `results/binary-cycle/dataset_statistics.json`. Tables: `comparison_summary.csv`; full changes: `modification_effects.csv`; binary paired records: `paired_folds.csv`; checks: `verification.json`, `preservation_verified.json`. No PTC-MR binary outputs were generated.', ''])
    (ROOT / 'SLIDES.md').write_text('\n'.join(slides))
    notes_path = ROOT / 'NOTES.md'; notes = notes_path.read_text()
    notes = notes.replace('Final range remains 0..6 if timings favor completion over extension; any extension must be decided without choosing by outer accuracy.', 'The completed final range is 0..6; it was not chosen by outer accuracy.')
    notes = notes.replace('Full run uses four independent worker processes', 'The corrected completed full run used eight independent worker processes (earlier preflight used four)')
    notes = notes.replace('Fresh graph-local feature/persistence comparisons and the documented smoke command will be run after dependency installation; they must pass before delivery.', 'Fresh graph-local feature/persistence comparisons and the corrected smoke validation passed; evidence remains in the preserved loader/validation records and corrected smoke outputs.')
    heading = '## Final case-study completion: binary cycle representation'
    if heading in notes: notes = notes.split(heading)[0].rstrip()+'\n\n'
    notes += f'''{heading}

MUTAG was selected as a small node-labeled molecular mutagenicity benchmark enabling an affordable controlled 100-fold follow-up; PTC-MR supplies a larger molecular carcinogenicity benchmark with different label balance and already-completed secondary results. Neither is chosen because an extension is guaranteed to win. Archive-verified statistics are in `results/binary-cycle/dataset_statistics.json`.

Shervashidze et al. (2011) supplies iterative neighborhood refinement and explicit subtree-pattern counts/induced kernels. Rieck et al. (2019), Algorithm 1 and §2.3.2, supplies persistent component and cycle features. We chose these papers because their shared WL refinement permits controlled topological feature comparisons. P-WL-C is an established paper-derived enhancement and a core implementation here; it is not newly invented in this project. Source-specific caps, root/edge ties and the evaluation differences above are explicit adaptations.

λ weighting was investigated to control relative cycle contribution. {lambda_text} It remains an attempted custom modification with ineffective positive-magnitude results, not erased evidence.

Binary asks whether presence rather than magnitude is useful. `pwl.combine_binary` transforms the final aggregated cycle coordinates only. Vertex values are unchanged; all zero keys and ordering are retained. Existing corrected h=6 p=1/2 caches were read directly; no persistence, preprocessing, old CV, or PTC-MR binary computation ran. `binary_cycle.py` invokes the unchanged generic evaluator with only the new method and the original saved MUTAG folds. No globally fitted feature matrix is reused; training vocabularies remain split-local. Resume guards include input, code and environment hashes. Five regression tests cover values/zeros, dimensions/order, vertex identity, cache reuse, original source functions and checkpoint behavior; original scientific checks also passed.

**Observed outcomes**

{impact_table(mutag)}

PTC-MR cycle versus P-WL: −0.55210 pp accuracy / −0.82427 pp macro-F1. {observed} {changed}

{interpretation}

Both original datasets and the MUTAG binary follow-up are complete. PTC-MR-Binary: **NOT RUN — OUT OF SCOPE; Not evaluated (out of scope)**. Preservation verification covers 712 existing result/cache files. {limits}

Further work could compare an independently held-out benchmark or a different classifier sensitive to magnitude; neither was performed or selected to obtain an improvement. Complete evidence: RESULTS.md, the single 17-slide SLIDES.md, `results/binary-cycle/verification.json` and paired CSVs.
'''
    notes_path.write_text(notes)
    log_path = ROOT / 'LOG.md'; log = log_path.read_text()
    log = log.replace('isolated dependency installation is in progress.', 'isolated dependency installation subsequently completed (historical Stage 0 status).')
    log = log.replace('## Stage 4 — running\nStarting the documented full command. No final result is claimed until all 200 outer folds, summaries, figures and consistency checks complete.', '## Stage 4 — completed corrected full run\nBoth corrected datasets completed all 100 outer folds, summaries, figures and consistency checks. The launch/interim entries below are historical; compact-cycle scores are superseded.')
    log = log.replace('### Stage 4 interim: MUTAG completed and TUDataset requirement verified', '### Superseded compact-column Stage 4 interim — historical scores only')
    log = log.replace('Final PTC-MR results remain pending.', 'PTC-MR was pending in that archived attempt; the corrected run subsequently completed all 100 folds. These cycle scores are excluded from final results.')
    log = log.replace('but final comparisons will use a corrected run.', 'and final comparisons use the completed corrected run.')
    log = log.replace('and will validate exact 2× column counts and repeat all cycle/extension checks before the corrected full run.', 'and validated exact 2× column counts and repeated all cycle/extension checks before the corrected full run.')
    log = log.replace('### Corrected Stage 2/3 checks passed; corrected full run started', '### Corrected Stage 2/3 checks passed; corrected full run completed')
    heading = '## Final binary follow-up — completed'
    if heading in log: log = log.split(heading)[0].rstrip()+'\n\n'
    log += f'''\n{heading}

Corrected original MUTAG and PTC-MR: 100/100 folds each, eight outer workers. Their metrics were independently recomputed without retraining. MUTAG baseline accuracies: WL 88.19%, P-WL 86.17%, P-WL-C 87.77%, λ-selected 87.77%. PTC-MR: WL 62.33%, P-WL 62.77%, P-WL-C 62.22%, λ-selected 62.49%. Original immutable evidence remains in `results/full/`; archived compact PTC-MR has 39/100 folds, 61 missing, and remains superseded.

Implemented `pwl.combine_binary` without changing existing functions. Command executed: `.venv/bin/python -B binary_cycle.py --workers 8 --output results/binary-cycle`. MUTAG only: 100/100 outer folds, 4,200 inner candidate means, original saved outer/inner splits and tree/depth/p grid. Checkpoints: `results/binary-cycle/checkpoints/MUTAG-r*-f*.json`. Cached corrected features reused, persistence recomputed: false. Predictions/parameters: `MUTAG_folds.csv`. {observed}

Validation: five tests in `test_binary.py` passed; original scientific checks passed (47 source pairings, toy GUDHI, complete alphabet, split-local vocabulary, independent forest prefixes). Saved metrics, repeat aggregation, inner selection and checkpoint agreement passed the final analysis audit. 712 pre-existing result/cache files and original method functions are unchanged, recorded in `preservation_before.json` / `preservation_verified.json`. Test outputs: `validation.log` and `scientific_checks.log` in the new binary directory.

Paired result: {counts.replace(chr(10),' ')} {changed} Paired CSV and comparison JSON retain every fold and full precision. New four 300 dpi figures are under `results/binary-cycle/figures/`; original audited figures were preserved. README, RESULTS, DEMO, NOTES and this LOG are updated; SLIDES contains exactly the required 17 slides with core content, visuals and presenter notes. Completed analysis outputs: `comparison_summary.csv`, `modification_effects.csv`, `paired_folds.csv`, `paired_repeat_means.csv`, `comparison.json`, `verification.json`.

**PTC-MR-Binary: NOT RUN — OUT OF SCOPE; Not evaluated (out of scope).** No PTC-MR binary job, command, checkpoint, test experiment, or figure was created. The λ findings remain visible and binary is the primary custom representation modification. No significance or universal improvement claim is made.
'''
    log_path.write_text(log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/binary-cycle')
    args = parser.parse_args(); out = args.output.resolve()
    assert out != (ROOT / 'results/full').resolve()
    _, _, _, _, cache_hashes = read_inputs()
    old = ROOT / 'results/full'
    original, _, mutag_summary, mutag_stats = audit_saved('MUTAG', old)
    secondary, _, ptc_summary, ptc_stats = audit_saved('PTC_MR', old)
    binary, _, binary_summary, _ = audit_saved('MUTAG', out, binary=True)
    pairs, info = compare(original, binary)
    summary = pd.concat([mutag_summary, binary_summary], ignore_index=True)
    summary.to_csv(out / 'comparison_summary.csv', index=False)
    pairs.to_csv(out / 'paired_folds.csv', index=False)
    pairs.groupby('repeat')[['accuracy_difference_pp', 'macro_f1_difference_pp']].mean().reset_index().to_csv(out / 'paired_repeat_means.csv', index=False)
    effects = []
    for dataset, stats in [('MUTAG', summary), ('PTC_MR', ptc_summary)]:
        s = stats.set_index('method')
        for a,b in [('WL','P-WL'), ('P-WL','P-WL-C'), ('P-WL-C','Extension'), ('P-WL-C',METHOD)]:
            if b in s.index:
                effects.append(dict(dataset=dataset, source_method=a, target_method=b,
                    accuracy_difference_pp=100*(s.loc[b,'accuracy_mean']-s.loc[a,'accuracy_mean']),
                    macro_f1_difference_pp=100*(s.loc[b,'macro_f1_mean']-s.loc[a,'macro_f1_mean'])))
    pd.DataFrame(effects).to_csv(out / 'modification_effects.csv', index=False)
    lambdas = {'MUTAG': lambda_audit(original), 'PTC_MR': lambda_audit(secondary)}
    (out / 'comparison.json').write_text(json.dumps(info, indent=2))
    (out / 'lambda_verification.json').write_text(json.dumps(lambdas, indent=2))
    stats = {'MUTAG': mutag_stats, 'PTC_MR': ptc_stats}
    (out / 'dataset_statistics.json').write_text(json.dumps(stats, indent=2))
    preserved = preservation(out)
    make_figures(summary, pairs, out)
    write_docs(summary, ptc_summary, stats, info, lambdas, out)
    from PIL import Image
    images = list((out / 'figures').glob('*.png')); assert len(images) == 4
    for path in images:
        with Image.open(path) as im: assert im.width >= 1800 and abs(im.info['dpi'][0]-300) < 1
    assert len([line for line in (ROOT / 'SLIDES.md').read_text().splitlines() if line.startswith('## Slide ')]) == 17
    for name, digest in cache_hashes.items(): assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    preservation(out)
    verification = dict(mutag_binary_folds=100, missing_folds=0, binary_inner_rows=4200,
        saved_metrics_recomputed=True, original_baselines_recomputed=True, nested_selection_verified=True,
        checkpoint_agreement=True, repeat_aggregation=True, paired_identical_splits=True,
        lambda_audit=lambdas, figures_300dpi=4, slides=17, ptc_binary='NOT RUN — OUT OF SCOPE', **preserved)
    (out / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(markdown_table(summary)); print(json.dumps(info, indent=2)); print('PASS: verified full case study; original results and caches preserved.')


if __name__ == '__main__':
    main()
