"""Controlled binary representation and baseline-preservation regression tests."""
import hashlib
import inspect
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.feature_extraction import DictVectorizer

import evaluate
import pwl
import wl
from binary_cycle import read_inputs, ROOT, METHOD


class BinaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y, cls.folds, cls.blocks, _, cls.hashes = read_inputs()

    def test_transform_all_cached_mutag_depths_and_p(self):
        for block in self.blocks.values():
            before = pickle.dumps(block)
            for h in range(7):
                original = pwl.combine(block, h, 1)
                binary = pwl.combine_binary(block, h)
                self.assertEqual(binary, pwl.combine_binary(block, h))
                for a, b in zip(original, binary):
                    self.assertEqual(list(a), list(b))
                    for key, value in a.items():
                        self.assertEqual(b[key], float(value != 0) if ':cycle:' in key else value)
                        if ':cycle:' in key:
                            self.assertIn(b[key], (0, 1))
                self.assertEqual(pwl.combine(block, h, 0), wl.cumulative(block['vertices'], h))
            self.assertEqual(before, pickle.dumps(block))

    def test_dimensions_order_and_vertex_matrix(self):
        for block in self.blocks.values():
            for h in range(7):
                a, b = pwl.combine(block, h, 1), pwl.combine_binary(block, h)
                for split in [self.folds[0], self.folds[-1]]:
                    # Outer and inner training vocabularies; held-out values never fit vocabularies.
                    for train, test in [(split['train'], split['test']), *split['inner'][:1]]:
                        v, w = DictVectorizer(dtype=np.float32), DictVectorizer(dtype=np.float32)
                        X = v.fit_transform([a[i] for i in train]); Z = w.fit_transform([b[i] for i in train])
                        self.assertEqual(v.feature_names_, w.feature_names_)
                        self.assertEqual(X.shape, Z.shape)
                        vertices = [i for i, key in enumerate(v.feature_names_) if ':cycle:' not in key]
                        self.assertEqual((X[:, vertices] != Z[:, vertices]).nnz, 0)
                        A, B = v.transform([a[i] for i in test]), w.transform([b[i] for i in test])
                        self.assertEqual((A[:, vertices] != B[:, vertices]).nnz, 0)

    def test_cache_reuse_without_persistence_or_loader_calls(self):
        with patch('pwl.persistence', side_effect=AssertionError('Persistence must not run')), \
             patch('pwl.pwl_features', side_effect=AssertionError('Cache generator must not run')):
            _, _, blocks, _, hashes = read_inputs()
            self.assertEqual(hashes, self.hashes)
            self.assertEqual(blocks, self.blocks)
        for name, digest in self.hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_existing_method_implementations_unchanged(self):
        evidence = ROOT / 'results/binary-cycle/preservation_before.json'
        self.assertTrue(evidence.exists(), 'Initial preservation evidence is required')
        hashes = json.loads(evidence.read_text())['functions']
        modules = {'wl': wl, 'pwl': pwl, 'evaluate': evaluate}
        for qualified, expected in hashes.items():
            module, name = qualified.split('.')
            source = inspect.getsource(getattr(modules[module], name))
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected, qualified)

    def test_checkpoint_resume_skips_fits_and_rejects_changed_inputs(self):
        split = dict(repeat=0, fold=0, train=[0, 1], test=[2], inner=[])
        rows = {(METHOD, 0, 1, 1): [{'a': 1}, {'a': 2}, {'a': 3}]}
        record = dict(dataset='MUTAG', method=METHOD, setting='selected', repeat=0, fold=0,
                      h=0, p=1, l=1, trees=25, predictions='[0]', test_indices='[2]',
                      accuracy=1., macro_f1=1., inner_accuracy=1.)
        y = np.array([0, 1, 0])
        inner = dict(dataset='MUTAG', method=METHOD, repeat=0, fold=0,
                     h=0, p=1, l=1, trees=25, inner_accuracy=1.)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with patch('evaluate.evaluate_fold', return_value=([record], [inner])) as fit:
                first, _ = evaluate.evaluate('MUTAG', y, rows, [split], [25], out)
                self.assertEqual(fit.call_count, 1)
            with patch('evaluate.evaluate_fold', side_effect=AssertionError('Completed folds must not refit')):
                second, _ = evaluate.evaluate('MUTAG', y, rows, [split], [25], out)
                self.assertEqual(first.predictions.tolist(), second.predictions.tolist())
                with self.assertRaisesRegex(AssertionError, 'Configuration changed'):
                    evaluate.evaluate('MUTAG', y, rows, [split], [50], out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
