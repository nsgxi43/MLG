
from collections import Counter

def wl_features(graph, node_labels, H=2):
    labels = dict(node_labels)
    features = Counter()

    for h in range(H + 1):
        
        for label in labels.values():
            features[(h, label)] += 1

        if h == H:
            break

     
        signatures = {}
        for v in graph:
            neighbor_labels = tuple(
                sorted(labels[u] for u in graph[v])
            )
            signatures[v] = (labels[v], neighbor_labels)

        # Assign the same new label to identical signatures
        unique = sorted(set(signatures.values()))
        encoding = {sig: i for i, sig in enumerate(unique)}
        labels = {
            v: encoding[signatures[v]]
            for v in graph
        }

    return features
