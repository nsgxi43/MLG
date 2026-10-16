"""Shared saved nested folds and train-only feature vocabularies."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


def fixed_folds(y, name, repeats, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{name}-r{repeats}.json'
    fingerprint = hashlib.sha256(np.asarray(y).tobytes()).hexdigest()
    if path.exists():
        payload = json.loads(path.read_text())
        assert payload['y_sha256'] == fingerprint
        return payload['folds']
    folds = []
    for repeat in range(repeats):
        outer = StratifiedKFold(10, shuffle=True, random_state=repeat)
        for fold, (train, test) in enumerate(outer.split(np.zeros(len(y)), y)):
            inner = StratifiedKFold(5, shuffle=True, random_state=1000 + 10 * repeat + fold)
            splits = [[train[a].tolist(), train[b].tolist()] for a, b in inner.split(train, y[train])]
            assert not set(train) & set(test)
            for a, b in splits:
                assert not set(a) & set(b) and set(a) | set(b) == set(train)
                assert not (set(a) | set(b)) & set(test)
            folds.append(dict(repeat=repeat, fold=fold, train=train.tolist(), test=test.tolist(), inner=splits))
    path.write_text(json.dumps(dict(y_sha256=fingerprint, folds=folds)))
    return folds


def matrices(rows, train, test):
    vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
    X = vectorizer.fit_transform([rows[i] for i in train]).tocsc()
    Z = vectorizer.transform([rows[i] for i in test]).tocsr()
    names = vectorizer.feature_names_
    cycles = sum(':cycle:' in key for key in names)
    if cycles:
        assert 2*cycles == len(names), 'Cycle and vertex alphabets must match'
    X.eliminate_zeros()
    Z.eliminate_zeros()
    return X, Z


def forest(n):
    return RandomForestClassifier(n_estimators=n, criterion='gini', max_features='sqrt',
                                  bootstrap=True, max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, class_weight=None, random_state=42, n_jobs=1)


def score_rows(rows, y, train, test, trees):
    """One maximal forest supplies exact deterministic smaller-forest prefixes."""
    X, Z = matrices(rows, train, test)
    clf = forest(max(trees)).fit(X, y[train])
    probs = np.zeros((len(test), len(clf.classes_)))
    predictions = {}
    for n, estimator in enumerate(clf.estimators_, 1):
        probs += estimator.predict_proba(Z, check_input=False)
        if n in trees:
            predictions[n] = clf.classes_[probs.argmax(axis=1)]
    return predictions


def evaluate_fold(name, y, representations, split, trees, base=None):
    start = time.perf_counter()
    repeat, fold = split['repeat'], split['fold']
    scores, inner_records, records = {}, [], []
    candidates = list(representations)
    # Reuse only exactly identical representations: P-WL = Extension(l=0),
    # P-WL-C = Extension(l=1). Other positive weights are actually fitted.
    aliases = {}
    for key in candidates:
        if key[0] == 'Extension' and key[3] in (0, 1):
            aliases[key] = (('P-WL' if key[3] == 0 else 'P-WL-C'),) + key[1:]
            assert representations[key] == representations[aliases[key]]
        else:
            aliases[key] = key
    computed = {}
    predictions = {}
    if base is not None:
        for row in base['inner']:
            if row['method'] in ('WL','P-WL'):
                key = (row['method'],row['h'],row['p'],row['l'])
                computed.setdefault(key,{})[row['trees']] = row['inner_accuracy']
        for row in base['records']:
            if row['method'] in ('WL','P-WL'):
                key = (row['method'],row['h'],row['p'],row['l'],row['trees'])
                pred = np.asarray(json.loads(row['predictions']))
                assert json.loads(row['test_indices']) == split['test']
                if key in predictions:
                    assert np.array_equal(predictions[key],pred)
                predictions[key] = pred
        assert all(set(values)==set(trees) for values in computed.values())
    for key in candidates:
        source = aliases[key]
        if source not in computed:
            values = {n: [] for n in trees}
            for a, b in split['inner']:
                preds = score_rows(representations[source], y, a, b, trees)
                for n, pred in preds.items():
                    values[n].append(accuracy_score(y[b], pred))
            computed[source] = {n: float(np.mean(value)) for n, value in values.items()}
        for n, value in computed[source].items():
            scores[key + (n,)] = value
            inner_records.append(dict(dataset=name, repeat=repeat, fold=fold, method=key[0],
                                      h=key[1], p=key[2], l=key[3], trees=n, inner_accuracy=value))
    groups = {}
    for method in sorted({k[0] for k in candidates}):
        keys = [k for k in scores if k[0] == method]
        groups[(method, 'selected')] = keys
        if method in ('WL', 'P-WL', 'P-WL-C'):
            for h in sorted({k[1] for k in keys}):
                groups[(method, f'h={h}')] = [k for k in keys if k[1] == h]
        if method == 'Extension':
            for l in sorted({k[3] for k in keys}):
                groups[(method, f'l={l:g}')] = [k for k in keys if k[3] == l]
    for (method, setting), keys in groups.items():
        best = min(keys, key=lambda k: (-scores[k], k[1], k[2], k[4], k[3]))
        source = aliases[best[:4]] + (best[4],)
        if source not in predictions:
            predictions[source] = score_rows(representations[source[:4]], y, split['train'], split['test'], [source[4]])[source[4]]
        pred = predictions[source]
        records.append(dict(dataset=name, method=method, setting=setting, repeat=repeat, fold=fold,
                            h=best[1], p=best[2], l=best[3], trees=best[4],
                            accuracy=accuracy_score(y[split['test']], pred),
                            macro_f1=f1_score(y[split['test']], pred, average='macro', zero_division=0),
                            inner_accuracy=scores[best], test_indices=json.dumps(split['test']),
                            predictions=json.dumps(pred.tolist())))
    elapsed = time.perf_counter() - start
    for record in records:
        record['fold_evaluation_seconds'] = elapsed
    return records, inner_records


def evaluate(name, y, representations, folds, trees, out, workers=1, reuse_base=None, legacy_representations=None):
    """Parallel independent outer folds; all choices remain inside inner CV."""
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    signature = hashlib.sha256(pickle.dumps((y, representations, folds, trees, 'eval-v2'))).hexdigest()
    legacy_signature = None
    if reuse_base is not None:
        assert legacy_representations is not None
        legacy_signature = hashlib.sha256(pickle.dumps((y, legacy_representations, folds, trees, 'eval-v2'))).hexdigest()
    records, inner_records, pending = [], [], []
    start = time.perf_counter()
    for split in folds:
        path = checkpoint / f"{name}-r{split['repeat']}-f{split['fold']}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['signature'] == signature, 'Configuration changed: choose a new output directory'
            records.extend(saved['records'])
            inner_records.extend(saved['inner'])
        else:
            base = None
            if reuse_base is not None:
                old = reuse_base / 'checkpoints' / path.name
                if old.exists():
                    base = json.loads(old.read_text())
                    assert base['signature'] == legacy_signature, 'Prior inputs/folds/grid do not match'
            pending.append((split, path, base))
    def save(split, path, result):
        rows, inner = result
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(signature=signature, records=rows, inner=inner)))
        temp.replace(path)
        records.extend(rows)
        inner_records.extend(inner)
        print(f"{name}: repeat {split['repeat'] + 1}, fold {split['fold'] + 1} complete ({len(records)//len(rows)}/{len(folds)})", flush=True)
    if workers == 1:
        for split, path, base in pending:
            save(split, path, evaluate_fold(name, y, representations, split, trees, base))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate_fold, name, y, representations, split, trees, base): (split,path) for split,path,base in pending}
            for future in as_completed(futures):
                split, path = futures[future]
                save(split, path, future.result())
    df = pd.DataFrame(records).sort_values(['repeat','fold','method','setting'])
    df.to_csv(out / f'{name}_folds.csv', index=False)
    pd.DataFrame(inner_records).sort_values(['repeat','fold','method','h','p','l','trees']).to_csv(out / f'{name}_inner.csv', index=False)
    elapsed = time.perf_counter()-start
    (out / f'{name}_evaluation.json').write_text(json.dumps(dict(evaluation_seconds=elapsed, folds=len(folds),
                                                                workers=workers, newly_evaluated_folds=len(pending),
                                                                reused_base_folds=sum(base is not None for _,_,base in pending))))
    return df, elapsed


def summarize(df):
    repeat = df.groupby(['dataset', 'method', 'setting', 'repeat'])[['accuracy', 'macro_f1']].mean().reset_index()
    stats = repeat.groupby(['dataset', 'method', 'setting']).agg(
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', lambda x: x.std(ddof=0) if len(x) > 1 else np.nan),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', lambda x: x.std(ddof=0) if len(x) > 1 else np.nan),
        repeats=('repeat', 'count')).reset_index()
    return repeat, stats
