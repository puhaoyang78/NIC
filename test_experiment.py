import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment import CLASSES, check_split, make_splits, prepare, statistics, load_signal_tensor, iter_batches, cache_directory, balanced_ids, frequency_statistics, aggregate_probabilities, metric, augment_signal, save_cm


class ExperimentTests(unittest.TestCase):
    def test_unequal_session_counts_and_leakage_rejection(self):
        rows = [dict(session=f'{c}/{i}.csv', label=c)
                for c, count in zip(CLASSES, [3, 2, 3, 4, 5, 2, 3, 3]) for i in range(count)]
        folds = make_splits(rows, 42)
        self.assertEqual(len(folds), 25)
        self.assertEqual(folds, make_splits(rows, 42))
        tested = [s for f in folds for s in f['test']]
        self.assertEqual(len(tested), 25)
        self.assertEqual(set(tested), {r['session'] for r in rows})
        for f in folds:
            self.assertEqual(len(f['test']), 1)
            for c in CLASSES:
                self.assertTrue(any(s.startswith(c + '/') for s in f['train']))
        broken = {k: list(v) for k, v in folds[0].items()}
        broken['val'].append(broken['test'][0])
        with self.assertRaisesRegex(AssertionError, 'Session leakage'):
            check_split(broken, rows)

    def test_energy_features(self):
        x = np.array([[1., -1., 1., -1.], [2., 2., 2., 2.]])
        np.testing.assert_allclose(statistics(x), [[1, 4, 1, 2, 0, 1, 1], [2, 16, 0, 0, 2, 2, 2]])

    def test_resident_batches_preserve_valid_windows_and_splits(self):
        import torch
        arrays = {'a': np.arange(24, dtype=np.float32).reshape(3, 8),
                  'b': np.arange(32, dtype=np.float32).reshape(4, 8) + 100}
        arrays['a'][1] = np.nan
        indices = {'a': np.array([0, 2]), 'b': np.array([1, 3])}
        signal, ids = load_signal_tensor(arrays, indices, torch.device('cpu'))
        targets = torch.tensor([0, 0, 1, 1])
        held_out = list(iter_batches(signal, targets, ids['b'], 3))
        self.assertEqual(len(held_out), 1)
        np.testing.assert_array_equal(held_out[0][0][:, 0].numpy(), arrays['b'][[1, 3]])
        self.assertEqual(held_out[0][1].tolist(), [1, 1])
        selected = torch.tensor([3, 0, 2])
        def shuffled():
            batches = list(iter_batches(signal, targets, selected, 2, torch.Generator().manual_seed(42)))
            self.assertEqual([len(y) for _, y in batches], [2, 1])
            return torch.cat([x for x, _ in batches]), torch.cat([y for _, y in batches])
        first, second = shuffled(), shuffled()
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        self.assertEqual(sorted(first[0][:, 0, 0].tolist()), sorted(signal[selected, 0, 0].tolist()))
        with self.assertRaisesRegex(ValueError, 'Nonfinite values in valid cached windows'):
            load_signal_tensor(arrays, {'a': np.array([1]), 'b': np.array([1])}, torch.device('cpu'))

    def test_balanced_sampling_only_uses_training_sessions(self):
        rows = [dict(session=f'{c}/{i}.csv', label=c)
                for c, count in zip(CLASSES, [3, 2, 3, 4, 5, 2, 3, 3]) for i in range(count)]
        ids, offset = {}, 0
        for i, row in enumerate(rows):
            count = 3 + i * 7
            ids[row['session']] = np.arange(offset, offset + count)
            offset += count
        label_map = {r['session']: CLASSES.index(r['label']) for r in rows}
        for split in make_splits(rows, 42):
            for binary in (True, False):
                draws, audit = balanced_ids(split, ids, label_map, binary, 123)
                np.testing.assert_array_equal(draws, balanced_ids(split, ids, label_map, binary, 123)[0])
                allowed = set(np.concatenate([ids[s] for s in split['train']]))
                self.assertTrue(set(draws) <= allowed)
                quotas = []
                for c in CLASSES:
                    counts = [r['draws'] for r in audit if r['label'] == c]
                    self.assertLessEqual(max(counts) - min(counts), 1)
                    quotas.append(sum(counts))
                if binary:
                    self.assertEqual(quotas[5], len(draws) // 2)
                    self.assertEqual(len(set(quotas[:5] + quotas[6:])), 1)
                else:
                    self.assertEqual(len(set(quotas)), 1)

    def test_frequency_features_and_augmentation(self):
        import torch
        t = np.arange(50000) / 100000
        x = np.stack([np.sin(2 * np.pi * 12000 * t), 3 * np.sin(2 * np.pi * 12000 * t), np.ones(len(t))])
        f, names = frequency_statistics(x, 100000)
        self.assertTrue(np.isfinite(f).all())
        self.assertAlmostEqual(f[0, names.index('dominant_frequency_hz')], 12000, delta=30)
        self.assertAlmostEqual(f[0, names.index('spectral_centroid_hz')], 12000, delta=30)
        np.testing.assert_allclose(f[:2, 1:12:2].sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(f[1, 0:12:2], 9 * f[0, 0:12:2], atol=1e-6)
        np.testing.assert_allclose(f[0, 1:12:2], f[1, 1:12:2], atol=1e-6)
        self.assertTrue((f[2] == 0).all())
        signal = torch.tensor(x[:, None], dtype=torch.float32)
        original = signal.clone()
        a = augment_signal(signal, torch.Generator().manual_seed(7))
        b = augment_signal(signal, torch.Generator().manual_seed(7))
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(signal, original)
        self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(torch.equal(a, signal))

    def test_aggregation_gaps_empty_and_binary_metrics(self):
        indices = np.array([0, 1, 2, 4, 5, 6, 7])
        p = np.column_stack([np.arange(7) / 10, 1 - np.arange(7) / 10])
        ends, averaged = aggregate_probabilities(indices, p, 3)
        self.assertEqual(ends.tolist(), [2, 5, 6])
        np.testing.assert_allclose(averaged, [p[:3].mean(0), p[3:6].mean(0), p[4:7].mean(0)])
        for size in (1, 3, 5, 10):
            ends, q = aggregate_probabilities(np.arange(12), np.tile([.2, .8], (12, 1)), size)
            self.assertEqual(len(ends), 13 - size)
            np.testing.assert_allclose(q, np.tile([.2, .8], (len(ends), 1)))
        ends, q = aggregate_probabilities(indices, p, 10)
        self.assertEqual(len(q), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty_confusion.csv'
            save_cm(path, np.array([], dtype=int), np.array([], dtype=int), CLASSES)
            import csv
            with path.open() as f:
                matrix = list(csv.reader(f))
            self.assertEqual(len(matrix), 9)
            self.assertTrue(all(int(v) == 0 for row in matrix[1:] for v in row[1:]))
        self.assertTrue(all(v is None for v in metric(np.array([], dtype=int), q, 2).values()))
        y = np.array([0, 0, 1, 1])
        q = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
        scores = metric(y, q, 2)
        self.assertTrue(all(v == 1 for v in scores.values()))
        scores = metric(y[:2], q[:2], 2)
        self.assertIsNone(scores['auroc'])
        self.assertIsNone(scores['auprc'])
        self.assertIsNone(scores['attack_recall'])
        self.assertEqual(scores['normal_recall'], 1)

    def test_four_durations_keep_identical_25_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, root = Path(tmp) / 'out', Path(tmp) / 'data'
            out.mkdir()
            rows = []
            fs = 1024
            t = np.arange(15 * fs + 17) / fs
            for c, count in zip(CLASSES, [3, 2, 3, 4, 5, 2, 3, 3]):
                (root / c).mkdir(parents=True)
                for i in range(count):
                    path = root / c / f'{i}.csv'
                    np.savetxt(path, np.column_stack([t, np.sin(2 * np.pi * 100 * t)]),
                               delimiter=',', fmt='%.8f', header='时间,通道 B\n(s),(V)\n', comments='')
                    rows.append(dict(session=str(path.relative_to(root)), label=c, bytes=path.stat().st_size,
                                     samples=len(t), sample_rate_hz=fs, time_start_s=0))
            (out / 'inventory.json').write_text(json.dumps(dict(data_root=str(root), sessions=rows)))
            reference = make_splits(rows, 42)
            for duration in (.5, 1, 2, 5):
                args = argparse.Namespace(output=str(out), seed=42, smoke=False,
                                          window_seconds=duration, target_rate=fs)
                prepare(args)
                config = json.loads((cache_directory(args) / 'dataset.json').read_text())
                self.assertEqual(config['folds'], reference)
                self.assertEqual(len(config['folds']), 25)
                for row in config['sessions']:
                    self.assertEqual(np.load(row['cache_file']).shape,
                                     (len(t) // round(duration * fs), round(duration * fs)))

    def test_full_prepare_tail_and_timestamp_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, root = Path(tmp) / 'out', Path(tmp) / 'data'
            out.mkdir()
            rows = []
            n = 4057
            time = np.arange(n) / 10000
            for c in CLASSES:
                (root / c).mkdir(parents=True)
                for i in range(2):
                    path = root / c / f'{i}.csv'
                    np.savetxt(path, np.column_stack([time, np.sin(2 * np.pi * 100 * time)]),
                               delimiter=',', fmt='%.8f', header='时间,通道 B\n(s),(V)\n', comments='')
                    rows.append(dict(session=str(path.relative_to(root)), label=c, bytes=path.stat().st_size,
                                     samples=n, sample_rate_hz=10000, time_start_s=0))
            (out / 'inventory.json').write_text(json.dumps(dict(data_root=str(root), sessions=rows)))
            args = argparse.Namespace(output=str(out), seed=42, smoke=False, window_seconds=.1, target_rate=5000)
            prepare(args)
            config = json.loads((cache_directory(args) / 'dataset.json').read_text())
            for r in config['sessions']:
                self.assertEqual(r['windows'], 4)
                self.assertEqual(r['dropped_tail_samples'], 57)
                self.assertEqual(np.load(r['cache_file']).shape, (4, 500))
            path = root / rows[0]['session']
            # Bad points straddle a window boundary; a bad tail point is counted too.
            signal = np.sin(2 * np.pi * 100 * time)
            signal[[999, 1000, 4001]] = [np.nan, np.inf, np.nan]
            np.savetxt(path, np.column_stack([time, signal]), delimiter=',', fmt='%.8f',
                       header='时间,通道 B\n(s),(V)\n', comments='')
            rows[0]['bytes'] = path.stat().st_size
            (out / 'inventory.json').write_text(json.dumps(dict(data_root=str(root), sessions=rows)))
            prepare(args)
            config = json.loads((cache_directory(args) / 'dataset.json').read_text())
            first = config['sessions'][0]
            self.assertEqual(first['invalid_signal_samples'], 3)
            self.assertEqual(first['excluded_window_indices'], [0, 1])
            self.assertEqual(first['candidate_windows'], 4)
            self.assertEqual(first['windows'], 2)
            cached = np.load(first['cache_file'])
            self.assertTrue(np.isnan(cached[:2]).all())
            from scipy.signal import resample_poly
            expected = resample_poly(signal[2000:4000].reshape(2, 1000), 1, 2, axis=1)
            np.testing.assert_allclose(cached[2:], expected, atol=1e-7)
            content = path.read_text().replace('0.10000000,', '0.11000000,', 1)
            path.write_text(content)
            with self.assertRaisesRegex(ValueError, 'Timestamp gap'):
                prepare(args)


if __name__ == '__main__':
    unittest.main()
