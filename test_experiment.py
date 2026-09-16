import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment import CLASSES, check_split, make_splits, prepare, statistics, load_signal_tensor, iter_batches, cache_directory, balanced_ids, frequency_statistics, aggregate_probabilities, metric, augment_signal, save_cm, make_unseen_splits, task_definition, BEHAVIORS


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

    def test_unseen_protocol_and_behavior_sampling(self):
        rows = [dict(session=f'{c}/{i}.csv', label=c)
                for c, count in zip(CLASSES, [3, 2, 3, 4, 5, 2, 3, 3]) for i in range(count)]
        folds, metadata = make_unseen_splits(rows, 42)
        self.assertEqual(len(folds), 14)
        self.assertEqual((folds, metadata), make_unseen_splits(rows, 42))
        label_map = {r['session']: CLASSES.index(r['label']) for r in rows}
        ids = {r['session']: np.arange(i * 10, i * 10 + 10) for i, r in enumerate(rows)}
        for split, meta in zip(folds, metadata):
            check_split(split, rows)
            unknown = {r['session'] for r in rows if r['label'] == meta['held_out_attack']}
            self.assertTrue(unknown <= set(split['test']))
            self.assertFalse(unknown & set(split['train'] + split['val']))
            self.assertEqual(sum(s.startswith('normal/') for s in split['test']), 1)
            selected, audit = balanced_ids(split, ids, label_map, True, 42)
            self.assertEqual(sum(r['draws'] for r in audit if r['label'] == 'normal'), len(selected) // 2)
            counts = [sum(r['draws'] for r in audit if r['label'] == c)
                      for c in CLASSES if c not in ('normal', meta['held_out_attack'])]
            self.assertEqual(len(set(counts)), 1)
        names, mapping = task_definition('behavior')
        self.assertEqual(names, BEHAVIORS)
        for split in make_splits(rows, 42):
            selected, audit = balanced_ids(split, ids, label_map, False, 42, task='behavior')
            quotas = [sum(r['draws'] for r in audit if mapping[CLASSES.index(r['label'])] == c) for c in range(5)]
            self.assertEqual(len(set(quotas)), 1)
            self.assertEqual(sum(quotas), len(selected))

    def test_saved_prediction_validation(self):
        from rq_analysis import read_predictions
        import csv
        names, _ = task_definition('multiclass')
        rows = {'normal/a.csv': dict(label='normal', candidate_windows=4, excluded_window_indices=[2])}
        split = dict(train=[], val=[], test=['normal/a.csv'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pred.csv'
            def write(session='normal/a.csv', positions=(0, 1, 3)):
                with path.open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['session','start_window','end_window','true_label','predicted_label',
                                     *['probability_' + n for n in names]])
                    for i in positions:
                        writer.writerow([session,i,i,'normal','normal',0,0,0,0,0,1,0,0])
            write()
            data = read_predictions(path, split, rows, names, 'multiclass')
            self.assertEqual(data['normal/a.csv'][0].tolist(), [0, 1, 3])
            text = path.read_text()
            path.write_text(text.replace(',1,0,0', ',1.0000000000000002,0,0'))
            read_predictions(path, split, rows, names, 'multiclass')
            path.write_text(text.replace(',1,0,0', ',1.1,0,0'))
            with self.assertRaisesRegex(AssertionError, 'Invalid probabilities'):
                read_predictions(path, split, rows, names, 'multiclass')
            write(session='normal/train.csv')
            with self.assertRaisesRegex(AssertionError, 'Prediction sessions'):
                read_predictions(path, split, rows, names, 'multiclass')
            write(positions=(0, 1, 2))
            with self.assertRaises(AssertionError):
                read_predictions(path, split, rows, names, 'multiclass')

    def test_analysis_common_endpoints_and_compare(self):
        from rq_analysis import analyze, compare
        from experiment import write_csv, write_json
        rows = [dict(session=f'{c}/{i}.csv', label=c, candidate_windows=12,
                     excluded_window_indices=[5] if i == 0 else [], source_window_samples=1000)
                for c, count in zip(CLASSES, [3, 2, 3, 4, 5, 2, 3, 3]) for i in range(count)]
        folds = make_splits(rows, 42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'dataset.json'
            write_json(dataset, dict(sessions=rows, folds=folds, seed=42, target_window_samples=1000))
            source = root / 'signal'
            config = dict(dataset=str(dataset), models=['rf'], tasks=['multiclass'], window_seconds=1,
                          augmentation='signal', effective_batch_size=8, split_seed=42, effective_epochs=20,
                          learning_rate=.001, sampling='balanced', target_rate_hz=1000, checkpoint_selection='final',
                          smoke=True, numpy_version=np.__version__, torch_version='fixture')
            write_json(source / 'config.json', config)
            records = []
            for k, split in enumerate(folds):
                session = split['test'][0]
                row = next(r for r in rows if r['session'] == session)
                label = CLASSES.index(row['label'])
                predictions = []
                for i in range(12):
                    if i in row['excluded_window_indices']:
                        continue
                    q = np.zeros(8)
                    q[label], q[(label + 1) % 8] = ((.2, .8) if i == 0 else (.8, .2))
                    predictions.append(dict(session=session, start_window=i, end_window=i,
                        true_label=CLASSES[label], predicted_label=CLASSES[q.argmax()],
                        **{'probability_' + name: float(q[j]) for j, name in enumerate(CLASSES)}))
                write_csv(source / f'fold{k}_multiclass_rf_aggregate1_predictions.csv', predictions)
                records.append(dict(fold=k, task='multiclass', model='rf', aggregation=1))
            write_json(source / 'metrics.json', records)
            write_json(source / 'summary.json', [dict(task='multiclass', model='rf', aggregation=1,
                       test_windows=292, pooled=dict(accuracy=.6, macro_f1=.6))])
            analyze(argparse.Namespace(results=source, aggregations=(1, 3, 5, 10)))
            results = json.loads((source / 'analysis_a-1-3-5-10/summary.json').read_text())
            self.assertEqual(len(results), 4)
            self.assertEqual({r['common_windows'] for r in results}, {51})
            self.assertTrue(all(r['common_endpoint_metrics']['macro_f1'] == 1 for r in results))
            self.assertLess(results[0]['pooled']['accuracy'], 1)
            self.assertLess(results[-1]['valid_folds'], 25)
            # Paired ablation arithmetic and rejection of a mismatched epoch budget.
            for augmentation, f1 in [('signal', .7), ('none', .6)]:
                directory = root / ('compare_' + augmentation)
                write_json(directory / 'config.json', dict(config, models=['raw_cnn'], augmentation=augmentation))
                write_json(directory / 'summary.json', [dict(task='multiclass', model='raw_cnn', aggregation=1,
                           test_windows=292, pooled=dict(accuracy=f1, macro_f1=f1))])
            args = argparse.Namespace(results=[root / 'compare_signal', root / 'compare_none'], destination=root / 'comparison')
            compare(args)
            effects = json.loads((root / 'comparison/augmentation_effects.json').read_text())
            self.assertAlmostEqual(effects[0]['delta_signal_minus_none']['macro_f1'], .1)
            changed = dict(config, models=['raw_cnn'], augmentation='none', effective_epochs=10)
            write_json(root / 'compare_none/config.json', changed)
            with self.assertRaisesRegex(AssertionError, 'effective_epochs'):
                compare(args)

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
        _, exact = aggregate_probabilities(indices, p, 1)
        np.testing.assert_array_equal(exact, p)
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
