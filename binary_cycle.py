"""MUTAG-only binary cycle experiment using the existing nested RF evaluator."""
import argparse
import hashlib
import io
import json
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np

from evaluate import evaluate, summarize
from pwl import VERSION, combine_binary
from plots import table

METHOD = 'P-WL-C-Binary'
ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(root=ROOT):
    """Read validated completed-run inputs without touching data/feature caches."""
    baseline = root / 'results/full'
    config = json.loads((baseline / 'config.json').read_text())
    assert json.loads((baseline / 'manifest.json').read_text())['complete']
    assert config['repeats'] == 10 and config['h_max'] == 6
    assert config['trees'] == [25, 50, 100] and config['p'] == [1, 2]
    metadata_path = baseline / 'MUTAG_data.json'
    metadata = json.loads(metadata_path.read_text())
    archive = root / 'cache/data/MUTAG/MUTAG.zip'
    assert sha(archive) == metadata['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        raw = np.loadtxt(io.BytesIO(z.read('MUTAG/MUTAG_graph_labels.txt')), dtype=int)
    y = np.unique(raw, return_inverse=True)[1]
    fold_path = root / 'cache/folds/MUTAG-r10.json'
    saved = json.loads(fold_path.read_text())
    assert hashlib.sha256(y.tobytes()).hexdigest() == saved['y_sha256']
    folds = saved['folds']
    assert {(s['repeat'], s['fold']) for s in folds} == {(r, f) for r in range(10) for f in range(10)}
    assert len(folds) == 100 and len(y) == metadata['graphs']
    for split in folds:
        train, test = set(split['train']), set(split['test'])
        assert not train & test and train | test == set(range(len(y)))
        assert len(split['inner']) == 5
        for a, b in split['inner']:
            assert not set(a) & set(b) and set(a) | set(b) == train
    blocks, hashes = {}, {str(p.relative_to(root)): sha(p) for p in [archive, fold_path, metadata_path, baseline / 'config.json']}
    for p in config['p']:
        path = root / 'cache/features' / f"{metadata['archive_sha256']}-{VERSION}-h6-p{p}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f'Required corrected persistence cache unavailable: {path}. No features were recomputed.')
        hashes[str(path.relative_to(root))] = sha(path)
        with path.open('rb') as stream:
            block = pickle.load(stream)
        assert len(block['vertices']) == len(block['cycles']) == 7
        for vertices, cycles in zip(block['vertices'], block['cycles']):
            assert len(vertices) == len(cycles) == len(y)
            for v, c in zip(vertices, cycles):
                assert {key.replace(':cycle:', ':') for key in c} == set(v)
                assert all(np.isfinite(value) and value >= 0 for value in c.values())
                assert all(np.isfinite(value) and value >= 1 for value in v.values())
        blocks[p] = block
    return y, folds, blocks, config, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/binary-cycle')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true', help='Resume the identical binary experiment only')
    args = parser.parse_args()
    out = args.output.resolve()
    assert args.workers > 0
    # Refuse destinations containing baseline artifacts or nested in old runs.
    for path in (ROOT / 'results').glob('*/config.json'):
        parent = path.parent.resolve()
        if json.loads(path.read_text()).get('method') != METHOD:
            assert out != parent and parent not in out.parents, 'Use an isolated binary output directory'
    y, folds, blocks, baseline_config, hashes = read_inputs()
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'pandas']}
    config = dict(method=METHOD, datasets=['MUTAG'], repeats=10, h_max=6, p=[1, 2],
                  trees=baseline_config['trees'], l=[1], representation_version='binary-cycle-v1',
                  input_sha256=hashes, versions=versions)
    # Include evaluator and transform code in resume identity, not worker count.
    config['code_sha256'] = {name: sha(ROOT / name) for name in ['pwl.py', 'wl.py', 'evaluate.py', 'binary_cycle.py']}
    path = out / 'config.json'
    if path.exists():
        assert args.resume, 'Output already has an experiment: use --resume or a fresh directory'
        assert json.loads(path.read_text()) == config, 'Inputs/code/environment changed: use a fresh directory'
        if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()).get('complete'):
            print('Already complete; preserving all existing binary outputs. Use finalize_case_study.py to rebuild analysis only.')
            return
    else:
        assert not args.resume, 'No configured binary experiment to resume'
        if out.exists():
            assert not list(out.glob('*_folds.csv')) and not (out / 'checkpoints').exists()
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))
    start = time.perf_counter()
    representations = {(METHOD, h, p, 1): combine_binary(blocks[p], h)
                       for h in range(7) for p in [1, 2]}
    df, elapsed = evaluate('MUTAG', y, representations, folds, config['trees'], out, workers=args.workers)
    repeats, stats = summarize(df)
    repeats.to_csv(out / 'MUTAG_repeat_means.csv', index=False)
    stats.to_csv(out / 'MUTAG_summary.csv', index=False)
    table(stats, out / 'MUTAG_summary.md')
    for name, expected in hashes.items():
        assert sha(ROOT / name) == expected, f'Input changed during evaluation: {name}'
    (out / 'manifest.json').write_text(json.dumps(dict(complete=True, dataset='MUTAG', outer_folds=len(folds),
        total_invocation_seconds=time.perf_counter()-start, evaluation_seconds=elapsed, workers=args.workers,
        persistence_recomputed=False, cache_inputs_unchanged=True, ptc_binary='NOT RUN — OUT OF SCOPE'), indent=2))
    print(stats.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
