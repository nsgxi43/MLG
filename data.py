"""Cached PyG TUDataset loading, preserving original TU filtration tie order."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import requests

URL = 'https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip'


def load_data(name, cache=Path('cache')):
    """Read original TU files, keeping node and edge order for persistence ties."""
    folder = cache / 'data' / name
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f'{name}.zip'
    if not archive.exists():
        response = requests.get(URL.format(name=name), timeout=90)
        response.raise_for_status()
        temp = archive.with_suffix('.tmp')
        temp.write_bytes(response.content)
        with zipfile.ZipFile(temp) as z:
            assert z.testzip() is None
        temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        def read(suffix, required=True):
            path = f'{name}/{name}_{suffix}.txt'
            if path not in z.namelist():
                if required:
                    raise ValueError(f'Missing {path}')
                return None
            return np.loadtxt(io.BytesIO(z.read(path)), delimiter=',', ndmin=1, dtype=int)
        ids = read('graph_indicator') - 1
        raw_y = read('graph_labels')
        labels = read('node_labels', False)
        edges = read('A').reshape(-1, 2) - 1
        available = {k: f'{name}/{name}_{k}.txt' in z.namelist() for k in
                     ('node_labels', 'edge_labels', 'node_attributes', 'edge_attributes')}
        edge_labels = read('edge_labels', False)
    graphs = [nx.Graph() for _ in raw_y]
    node_map = {}
    for global_id, graph_id in enumerate(ids):
        local_id = len(graphs[graph_id])
        node_map[global_id] = local_id
        label = 0 if labels is None else int(labels[global_id])
        graphs[graph_id].add_node(local_id, label=label)
    ordered_edges = [[] for _ in graphs]
    seen = [set() for _ in graphs]
    for u, v in edges:
        assert ids[u] == ids[v], 'Cross-graph edge'
        graph_id = ids[u]
        a, b = node_map[u], node_map[v]
        assert a != b, 'Self-loops are outside the verified method'
        edge = tuple(sorted((a, b)))
        if edge not in seen[graph_id]:
            seen[graph_id].add(edge)
            ordered_edges[graph_id].append(edge)
            graphs[graph_id].add_edge(*edge)
    for g, e in zip(graphs, ordered_edges):
        g.graph['edge_order'] = e
    classes, y = np.unique(raw_y, return_inverse=True)
    # Feed the cached original archive to TUDataset. Preserve raw edge order
    # because PyG coalesces/sorts edges and that changes persistence tie attribution.
    from torch_geometric.datasets import TUDataset
    raw_dir = cache / 'tudataset' / name / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.namelist():
            if entry.startswith(f'{name}/{name}_') and entry.endswith('.txt'):
                target = raw_dir / Path(entry).name
                if not target.exists():
                    target.write_bytes(z.read(entry))
    dataset = TUDataset(root=str(cache / 'tudataset'), name=name, cleaned=False,
                        use_node_attr=False, use_edge_attr=False)
    assert len(dataset) == len(graphs)
    loaded, loaded_y = [], []
    for i, sample in enumerate(dataset):
        assert sample.num_nodes == len(graphs[i])
        node_labels = sample.x.argmax(dim=1).tolist() if dataset.num_node_labels else [0]*sample.num_nodes
        label_min = 0 if labels is None else int(labels.min())
        node_labels = [label + label_min for label in node_labels]
        assert node_labels == [graphs[i].nodes[v]['label'] for v in graphs[i]]
        g = nx.Graph()
        g.add_nodes_from((v, {'label': label}) for v, label in enumerate(node_labels))
        g.add_edges_from(sample.edge_index.t().tolist())
        assert set(map(frozenset, g.edges())) == set(map(frozenset, graphs[i].edges()))
        g.graph['edge_order'] = ordered_edges[i]
        loaded.append(g)
        loaded_y.append(int(sample.y.item()))
    assert np.array_equal(loaded_y, y)
    graphs, y = loaded, np.asarray(loaded_y, dtype=y.dtype)
    assert len(graphs) == len(y) and all(len(g) > 0 for g in graphs)
    assert np.bincount(y).min() >= 10
    metadata = dict(dataset=name, graphs=len(graphs), classes=classes.tolist(),
                    class_counts={str(k): int(sum(raw_y == k)) for k in classes},
                    nodes=sum(map(len, graphs)), edges=sum(g.number_of_edges() for g in graphs),
                    node_label_values=[] if labels is None else np.unique(labels).tolist(),
                    edge_label_values=[] if edge_labels is None else np.unique(edge_labels).tolist(),
                    available_attributes=available,
                    components=sum(nx.number_connected_components(g) for g in graphs),
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), source=URL.format(name=name),
                    loader='torch_geometric.datasets.TUDataset', raw_identity_verified=True)
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return graphs, y, metadata
