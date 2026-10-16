"""Scientific correctness checks; callable through run.py --check."""
import importlib.util
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np

from evaluate import forest, matrices, score_rows
from pwl import combine, persistence, pwl_features, check_topology
from wl import cumulative, wl_features


def run_checks():
    # Counterexample structure: WL cannot distinguish a uniform six-cycle
    # from two disjoint uniform triangles; persistence sees one vs two cycles.
    graphs = [nx.cycle_graph(6), nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))]
    for g in graphs:
        nx.set_node_attributes(g, 0, 'label')
        g.graph['edge_order'] = list(g.edges())
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        f = wl_features(graphs, 2, 'toy', Path(folder))
        for h in range(3):
            rows = cumulative(f['wl'], h)
            assert rows[0] == rows[1]
    assert [len(persistence(g, [1.0]*6)[1]) for g in graphs] == [1, 2]
    tree = nx.path_graph(4)
    nx.set_node_attributes(tree,99,'label')
    tree.graph['edge_order'] = list(tree.edges())
    with tempfile.TemporaryDirectory() as folder:
        sample = graphs + [tree]
        f = wl_features(sample,2,'column-check',Path(folder))
        blocks = pwl_features(sample,f,1,'column-check',Path(folder))
        assert check_topology(sample,f,blocks,1)==9
        for h in range(3):
            A,_ = matrices(combine(blocks,h,0),[0,1,2],[0])
            B,_ = matrices(combine(blocks,h,1),[0,1,2],[0])
            assert B.shape[1]==2*A.shape[1]
    # Verify training vocabulary cannot be changed by held-out labels.
    rows = [{'known': 1}, {'known': 2}, {'known': 3, 'test-only': 10}]
    X, Z = matrices(rows, [0,1], [2])
    assert X.shape == (2,1) and Z.shape == (1,1)
    altered = rows[:2] + [{'known': 999, 'another-test-only': 40}]
    A, _ = matrices(altered, [0,1], [2])
    assert (X != A).nnz == 0
    rows = [{'u': float(i % 7), 'v': float(np.sqrt(i)), 'cycle:a': float(i % 5)} for i in range(60)]
    y = np.array([i % 3 for i in range(60)])
    a, b = list(range(40)), list(range(40,60))
    prefix = score_rows(rows, y, a, b, [25,50,100])
    X, Z = matrices(rows, a, b)
    for n in [25,50]:
        assert np.array_equal(prefix[n], forest(n).fit(X,y[a]).predict(Z))
    # Compare attribution and ties to unmodified authors' topology.py.
    # Only igraph's type annotation import is required; a tiny graph adapter
    # allows the original calculator to execute without installing igraph.
    sys.modules.setdefault('igraph', types.ModuleType('igraph'))
    spec = importlib.util.spec_from_file_location('reference_topology', 'references/topology.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Edge:
        def __init__(self, pair):
            self.tuple = pair

    class Edges:
        def __init__(self, pairs, weights):
            self.pairs, self.weights = pairs, weights
        def __getitem__(self, key):
            return self.weights if key == 'weight' else Edge(self.pairs[key])

    class Graph:
        def __init__(self, g, weights):
            self.g = g
            self.es = Edges(g.graph['edge_order'], weights)
        def vcount(self):
            return len(self.g)

    rng = np.random.default_rng(42)
    count = 0
    for n in range(3, 15):
        for seed in range(4):
            g = nx.gnp_random_graph(n, 0.4, seed=seed)
            if not g.number_of_edges():
                continue
            g.graph['edge_order'] = list(g.edges())
            weights = rng.integers(1,5,size=g.number_of_edges()).tolist()
            deaths, cycles = persistence(g, weights)
            D, C = module.PersistenceDiagramCalculator().fit_transform(Graph(g, weights))
            expected = np.zeros(n)
            for birth, death, v in D:
                expected[v] = death-birth
            assert np.array_equal(deaths, expected) and cycles == C
            count += 1
    print(f'PASS: WL counterexample, complete cycle alphabet, GUDHI toy diagrams, train-only vocabulary, 25/50-tree prefixes, {count} authors-source pairing comparisons', flush=True)
    return dict(author_pairing_comparisons=count, wl_counterexample=True, complete_cycle_alphabet=True, train_only_vocabulary=True, forest_prefixes=True)
