"""Graph-local WL signatures and subtree-count features."""
import hashlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path

VERSION = 'wl-v1'


def signature(value, registry):
    raw = json.dumps(value, separators=(',', ':'))
    key = hashlib.sha256(raw.encode()).hexdigest()
    if key in registry:
        assert registry[key] == raw, 'WL signature collision'
    registry[key] = raw
    return key


def wl_features(graphs, h_max, digest, cache=Path('cache')):
    path = cache / 'features' / f'{digest}-{VERSION}-h{h_max}.pkl'
    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = {}
    labels = []
    neighbors = []
    features = []
    times = []
    current = None
    for h in range(h_max + 1):
        start = time.perf_counter()
        if h == 0:
            current = [[signature(['initial', g.nodes[v]['label']], registry) for v in g] for g in graphs]
            counts = [[Counter() for _ in g] for g in graphs]
        else:
            counts = [[Counter(current[i][u] for u in g.neighbors(v)) for v in g] for i, g in enumerate(graphs)]
            current = [[signature([current[i][v], sorted(counts[i][v].elements())], registry)
                        for v in g] for i, g in enumerate(graphs)]
        labels.append(current)
        neighbors.append(counts)
        features.append([dict(Counter(f'{h}:{label}' for label in row)) for row in current])
        times.append(time.perf_counter() - start)
    result = dict(labels=labels, neighbors=neighbors, wl=features, wl_seconds=times)
    with path.open('wb') as f:
        pickle.dump(result, f)
    return result


def cumulative(blocks, h):
    """Concatenate iteration dictionaries with distinct feature keys."""
    rows = []
    for i in range(len(blocks[0])):
        row = {}
        for block in blocks[:h + 1]:
            row.update(block[i])
        rows.append(row)
    return rows
