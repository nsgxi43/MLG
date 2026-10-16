"""Persistent WL with separate component and cycle dictionaries."""
import pickle
import time
from pathlib import Path

import numpy as np

from wl import cumulative

VERSION = 'pwl-v2-full-cycle-alphabet'


def persistence(g, weights):
    """Authors' stable graph filtration and root attribution, without filling cycles."""
    edges = g.graph['edge_order']
    parent = list(range(len(g)))

    def find(v):
        root = v
        while parent[root] != root:
            root = parent[root]
        while parent[v] != v:
            old = parent[v]
            parent[v] = root
            v = old
        return root

    deaths = np.zeros(len(g))
    cycle_edges = []
    for index in np.argsort(weights, kind='stable'):
        u, v = edges[index]
        a, b = find(u), find(v)
        if a == b:
            cycle_edges.append(int(index))
        else:
            small, large = sorted((a, b))
            deaths[small] = weights[index]
            parent[small] = large
    roots = [v for v in g if parent[v] == v]
    cap = max(weights, default=1.0)
    deaths[roots] = cap
    assert len(cycle_edges) == len(edges) - len(g) + len(roots)
    assert np.isfinite(deaths).all() and (deaths >= 1).all()
    return deaths, cycle_edges


def edge_weights(g, features, graph_id, h, p):
    if h == 0:
        labels = features['labels'][0][graph_id]
        return [1.0 + (labels[u] != labels[v]) for u, v in g.graph['edge_order']]
    labels = features['labels'][h-1][graph_id]
    counts = features['neighbors'][h][graph_id]
    values = []
    for u, v in g.graph['edge_order']:
        A, B = counts[u], counts[v]
        total = sum(abs(A[k] - B[k])**p for k in A.keys() | B.keys())
        values.append(1.0 + (labels[u] != labels[v]) + total**(1.0/p))
    return values


def pwl_features(graphs, features, p, digest, cache=Path('cache')):
    h_max = len(features['labels']) - 1
    path = cache / 'features' / f'{digest}-{VERSION}-h{h_max}-p{p}.pkl'
    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, cycles, times = [], [], []
    for h in range(h_max+1):
        start = time.perf_counter()
        V, C = [], []
        for i, g in enumerate(graphs):
            weights = edge_weights(g, features, i, h, p)
            deaths, indices = persistence(g, weights)
            labels = features['labels'][h][i]
            vrow = {}
            # Both blocks share the node-label alphabet, even when all
            # cycle contributions for a training label are zero.
            crow = {f'{h}:cycle:{label}': 0.0 for label in labels}
            for v in g:
                key = f'{h}:{labels[v]}'
                vrow[key] = vrow.get(key, 0.0) + deaths[v]**p
            for index in indices:
                for v in g.graph['edge_order'][index]:
                    key = f'{h}:cycle:{labels[v]}'
                    crow[key] = crow.get(key, 0.0) + weights[index]**p
            V.append(vrow)
            C.append(crow)
        vertices.append(V)
        cycles.append(C)
        times.append(time.perf_counter()-start)
    result = dict(vertices=vertices, cycles=cycles, persistence_seconds=times)
    with path.open('wb') as f:
        pickle.dump(result, f)
    return result


def combine(blocks, h, l):
    """Zero weight has exactly the P-WL vocabulary and representation."""
    V = cumulative(blocks['vertices'], h)
    if l == 0:
        return V
    C = cumulative(blocks['cycles'], h)
    return [{**v, **{key: l*value for key, value in c.items()}} for v, c in zip(V, C)]


def combine_binary(blocks, h):
    """Keep P-WL unchanged; encode cycle presence without dropping zero keys.

    Transform the completed label-indexed cycle contributions, not individual
    edges. Iteration namespaces and dictionary insertion order are unchanged.
    Input blocks are never mutated and no persistence computation is performed.
    """
    rows = combine(blocks, h, 1)
    return [{key: (float(value != 0) if ':cycle:' in key else value)
             for key, value in row.items()} for row in rows]


def check_topology(graphs, features, blocks, p):
    """Compare H0 deaths and H1 births to independent GUDHI diagrams."""
    import gudhi
    checks = 0
    for h in range(len(features['labels'])):
        assert combine(blocks, h, 0) == cumulative(blocks['vertices'], h)
        for i, g in enumerate(graphs):
            weights = edge_weights(g, features, i, h, p)
            deaths, indices = persistence(g, weights)
            st = gudhi.SimplexTree()
            for v in g:
                st.insert([v], filtration=0)
            for (u, v), weight in zip(g.graph['edge_order'], weights):
                st.insert([u, v], filtration=weight)
            st.persistence(persistence_dim_max=True)
            D0 = st.persistence_intervals_in_dimension(0)
            expected = np.where(np.isinf(D0[:,1]), max(weights, default=1.0), D0[:,1])
            assert np.allclose(sorted(deaths), sorted(expected))
            D1 = st.persistence_intervals_in_dimension(1)
            assert len(D1) == len(indices)
            assert np.allclose(sorted(D1[:,0]), sorted(weights[j] for j in indices))
            assert np.isinf(D1[:,1]).all()
            checks += 1
    return checks
