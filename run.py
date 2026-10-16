"""Single entry point. All prespecified experiment settings are here."""
# Configuration: smoke validates the pipeline; full uses the paper's repeat count.
CONFIG = {'smoke': {'repeats': 1, 'h_max': 1, 'trees': [25]},
          'full': {'repeats': 10, 'h_max': 6, 'trees': [25, 50, 100]}}
P_VALUES = [1, 2]
L_VALUES = [0, 0.25, 0.5, 1, 2]  # Our extension: lambda, called l in Python.

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from data import load_data
from evaluate import evaluate, fixed_folds, summarize
from wl import cumulative, wl_features
from plots import figures, table
from pwl import check_topology, combine, pwl_features


def main():
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=CONFIG, default='smoke')
    parser.add_argument('--datasets', nargs='+', choices=['MUTAG', 'PTC_MR'], default=['MUTAG'])
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--h-max', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='Run scientific correctness checks and exit')
    parser.add_argument('--workers', type=int, default=4, help='Independent outer-fold worker processes')
    parser.add_argument('--report-only', action='store_true', help='Audit completed full results and regenerate measured reports')
    parser.add_argument('--reuse-base', type=Path, help='Reuse verified unchanged WL/P-WL fits from the archived compact-column attempt')
    args = parser.parse_args()
    if args.report_only:
        from report import write_reports
        write_reports(args.output or Path('results/full'))
        return
    if args.check:
        from checks import run_checks
        run_checks()
        return
    config = dict(CONFIG[args.mode])
    if args.repeats is not None:
        config['repeats'] = args.repeats
    if args.h_max is not None:
        config['h_max'] = args.h_max
    assert config['repeats'] > 0 and config['h_max'] >= 0
    out = args.output or Path('results') / args.mode
    out.mkdir(parents=True, exist_ok=True)
    used_p = P_VALUES if args.stage >= 2 else []
    used_l = L_VALUES if args.stage >= 3 else ([0] if args.stage == 2 else [])
    (out / 'config.json').write_text(json.dumps(dict(**config, p=used_p, l=used_l, stage=args.stage, datasets=args.datasets, workers=args.workers), indent=2))
    tables = []
    runtimes = []
    for name in args.datasets:
        graphs, y, metadata = load_data(name)
        (out / f'{name}_data.json').write_text(json.dumps(metadata, indent=2))
        features = wl_features(graphs, config['h_max'], metadata['archive_sha256'])
        representations = {('WL', h, 0, 0): cumulative(features['wl'], h) for h in range(config['h_max'] + 1)}
        import numpy as np
        for h, seconds in enumerate(np.cumsum(features['wl_seconds'])):
            runtimes.append(dict(dataset=name, method='WL', p=0, h=h, seconds=seconds))
        if args.stage >= 2:
            topology_checks = 0
            for p in P_VALUES:
                blocks = pwl_features(graphs, features, p, metadata['archive_sha256'])
                topology_checks += check_topology(graphs, features, blocks, p)
                for h in range(config['h_max'] + 1):
                    representations[('P-WL', h, p, 0)] = combine(blocks, h, 0)
                    representations[('P-WL-C', h, p, 1)] = combine(blocks, h, 1)
                    for l in L_VALUES if args.stage >= 3 else [0]:
                        representations[('Extension', h, p, l)] = combine(blocks, h, l)
                for h, seconds in enumerate(np.cumsum(np.array(features['wl_seconds']) + blocks['persistence_seconds'])):
                    for method in ['P-WL', 'P-WL-C']:
                        runtimes.append(dict(dataset=name, method=method, p=p, h=h, seconds=seconds))
            (out / f'{name}_topology_checks.json').write_text(json.dumps(dict(gudhi_comparisons=topology_checks, zero_representation_equal=True)))
        legacy = None
        if args.reuse_base is not None:
            import pickle
            assert args.stage == 4 and args.mode == 'full'
            legacy = {('WL',h,0,0): cumulative(features['wl'],h) for h in range(config['h_max']+1)}
            for p in P_VALUES:
                old_path = Path('cache/features') / f"{metadata['archive_sha256']}-pwl-v1-h{config['h_max']}-p{p}.pkl"
                with old_path.open('rb') as file:
                    old_blocks = pickle.load(file)
                new_blocks = pwl_features(graphs, features, p, metadata['archive_sha256'])
                assert old_blocks['vertices'] == new_blocks['vertices']
                for h in range(config['h_max']+1):
                    legacy[('P-WL',h,p,0)] = combine(old_blocks,h,0)
                    legacy[('P-WL-C',h,p,1)] = combine(old_blocks,h,1)
                    for l in L_VALUES:
                        legacy[('Extension',h,p,l)] = combine(old_blocks,h,l)
        folds = fixed_folds(y, name, config['repeats'], Path('cache/folds'))
        df, elapsed = evaluate(name, y, representations, folds, config['trees'], out, args.workers, args.reuse_base, legacy)
        if args.stage >= 2:
            pwl = df[(df.method == 'P-WL') & (df.setting == 'selected')]
            zero = df[(df.method == 'Extension') & (df.setting == ('l=0' if args.stage >= 3 else 'selected'))]
            assert pwl.predictions.tolist() == zero.predictions.tolist()
            assert np.array_equal(pwl[['accuracy','macro_f1']].values, zero[['accuracy','macro_f1']].values)
            (out / f'{name}_zero_check.json').write_text(json.dumps(dict(predictions_equal=True, metrics_equal=True, outer_folds=len(folds))))
        repeat, stats = summarize(df)
        repeat.to_csv(out / f'{name}_repeat_means.csv', index=False)
        stats.to_csv(out / f'{name}_summary.csv', index=False)
        tables.append(stats)
        print(stats[stats.setting == 'selected'].to_string(index=False), flush=True)
        if args.stage == 4:
            import tempfile
            measurements = []
            for trial in range(3):
                with tempfile.TemporaryDirectory() as folder:
                    timed = wl_features(graphs, config['h_max'], metadata['archive_sha256'], Path(folder))
                    for h, seconds in enumerate(np.cumsum(timed['wl_seconds'])):
                        measurements.append(dict(dataset=name, method='WL', p=0, h=h, trial=trial, seconds=seconds))
                    for p in P_VALUES:
                        timed_pwl = pwl_features(graphs, timed, p, metadata['archive_sha256'], Path(folder))
                        for h, seconds in enumerate(np.cumsum(np.array(timed['wl_seconds']) + timed_pwl['persistence_seconds'])):
                            for method in ['P-WL', 'P-WL-C']:
                                measurements.append(dict(dataset=name, method=method, p=p, h=h, trial=trial, seconds=seconds))
            pd.DataFrame(measurements).to_csv(out / f'{name}_runtime_trials.csv', index=False)
            runtimes = [r for r in runtimes if r['dataset'] != name]
            timing = pd.DataFrame(measurements).groupby(['dataset','method','p','h']).agg(
                seconds=('seconds','mean'), seconds_std=('seconds',lambda x:x.std(ddof=0))).reset_index()
            runtimes.extend(timing.to_dict('records'))
    stats = pd.concat(tables)
    stats.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(runtimes).to_csv(out / 'feature_runtime.csv', index=False)
    table(stats, out / 'summary.md')
    figures(stats, pd.DataFrame(runtimes), out)
    import importlib.metadata
    versions = {m: importlib.metadata.version(m) for m in ['numpy','scipy','scikit-learn','networkx','gudhi','matplotlib','pandas','requests']}
    (out / 'manifest.json').write_text(json.dumps(dict(total_run_seconds=time.perf_counter()-run_start,
                                                      versions=versions, complete=True), indent=2))
    if args.stage == 4 and args.mode == 'full' and config['repeats'] > 1:
        from report import write_reports
        write_reports(out)


if __name__ == '__main__':
    main()
