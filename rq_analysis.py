"""RQ2/RQ4 analysis of saved out-of-session probabilities; never trains a model."""
import csv
import json
from pathlib import Path

import numpy as np

from experiment import (CLASSES, aggregate_probabilities, check_split, make_splits,
                        make_unseen_splits, metric, save_cm, task_definition, write_csv, write_json)


def mean_metrics(records, keys):
    result = {}
    for key in keys:
        values = [r[key] for r in records if r[key] is not None]
        result[key] = dict(mean=float(np.mean(values)) if values else None,
                           std=float(np.std(values)) if values else None, count=len(values))
    return result


def read_predictions(path, split, rows, names, task):
    """Verify complete original held-out window coverage before any aggregation."""
    with path.open() as f:
        records = list(csv.DictReader(f))
    assert {r['session'] for r in records} == set(split['test']), 'Prediction sessions differ from test split'
    by_session = {}
    mapping = task_definition(task)[1]
    for session in split['test']:
        source = rows[session]
        selected = [r for r in records if r['session'] == session]
        indices = np.array([int(r['end_window']) for r in selected], dtype=np.int64)
        expected = np.setdiff1d(np.arange(source['candidate_windows']), source['excluded_window_indices'])
        np.testing.assert_array_equal(indices, expected)
        assert all(int(r['start_window']) == int(r['end_window']) for r in selected), 'Need single-window predictions'
        truth = mapping[CLASSES.index(source['label'])]
        assert all(r['true_label'] == names[truth] for r in selected), 'Label mismatch'
        prob = np.array([[float(r['probability_' + name]) for name in names] for r in selected])
        # Historical aggregate=1 exports used cumulative subtraction, causing ~1e-13 roundoff.
        assert np.isfinite(prob).all() and ((prob >= -1e-12) & (prob <= 1 + 1e-12)).all(), 'Invalid probabilities'
        np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-5)
        assert all(r['predicted_label'] == names[int(q.argmax())] for r, q in zip(selected, prob))
        by_session[session] = (indices, np.full(len(indices), truth, dtype=np.int64), prob)
    return by_session


def class_statistics(path, y, probabilities, names):
    from sklearn.metrics import precision_recall_fscore_support
    pred = probabilities.argmax(axis=1)
    save_cm(Path(str(path) + '_confusion.csv'), y, pred, names)
    cm = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    support = cm.sum(axis=1)
    normalized = cm / np.maximum(support[:, None], 1)
    write_csv(Path(str(path) + '_confusion_normalized.csv'), [dict(true_label=name,
              **{label: float(normalized[i, j]) for j, label in enumerate(names)}) for i, name in enumerate(names)])
    if len(y):
        precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=range(len(names)), zero_division=0)
    else:
        precision = recall = f1 = np.zeros(len(names))
    write_csv(Path(str(path) + '_per_class.csv'), [dict(label=name, support=int(support[i]),
              precision=float(precision[i]), recall=float(recall[i]) if support[i] else None,
              f1=float(f1[i])) for i, name in enumerate(names)])
    if names == CLASSES:
        pair = ['dirsearch', 'gobuster']
        write_json(Path(str(path) + '_dirsearch_gobuster.json'), {
            name: dict(support=int(support[names.index(name)]),
                       correct=int(cm[names.index(name), names.index(name)]),
                       confused_with_other_tool=int(cm[names.index(name), names.index(pair[1 - i])]),
                       predicted_outside_pair=int(cm[names.index(name)].sum() - cm[names.index(name), [names.index(p) for p in pair]].sum()),
                       destinations={label: int(cm[names.index(name), j]) for j, label in enumerate(names)})
            for i, name in enumerate(pair)})


def analyze(args):
    source = Path(args.results).resolve()
    config = json.loads((source / 'config.json').read_text())
    # Completion marker: refuse partially trained runs.
    json.loads((source / 'summary.json').read_text())
    dataset = json.loads(Path(config['dataset']).read_text())
    rows = {r['session']: r for r in dataset['sessions']}
    protocol = config.get('protocol', 'loso')
    if protocol == 'unseen':
        expected_folds, metadata = make_unseen_splits(dataset['sessions'], dataset['seed'])
    else:
        expected_folds = make_splits(dataset['sessions'], dataset['seed'])
        metadata = [{} for _ in expected_folds]
    folds = config.get('evaluation_folds', dataset['folds'])
    assert folds == expected_folds, 'Changed evaluation folds'
    for split in folds:
        check_split(split, dataset['sessions'])
    tasks = config.get('tasks', ['binary', 'multiclass'])
    models = config['models']
    sizes = sorted(set([1, *args.aggregations]))
    output = source / ('analysis_a-' + '-'.join(map(str, sizes)))
    output.mkdir(exist_ok=True)
    (output / 'summary.json').unlink(missing_ok=True)
    metrics = json.loads((source / 'metrics.json').read_text())
    keys = [(r['fold'], r['task'], r['model']) for r in metrics if r['aggregation'] == 1]
    expected = {(fold, task, model) for fold in range(len(folds)) for task in tasks for model in models}
    assert len(keys) == len(set(keys)) and set(keys) == expected, 'Incomplete single-window evaluation'
    fold_results, session_results, summary = [], [], []
    for task in tasks:
        names = config.get('task_names', {}).get(task, task_definition(task)[0])
        for model in models:
            pooled = {size: [] for size in sizes}
            matched = {size: [] for size in sizes}
            for fold_id, split in enumerate(folds):
                path = source / f'fold{fold_id}_{task}_{model}_aggregate1_predictions.csv'
                sessions = read_predictions(path, split, rows, names, task)
                fold_chunks = {size: [] for size in sizes}
                for session, (indices, y, prob) in sessions.items():
                    aggregated = {size: aggregate_probabilities(indices, prob, size) for size in sizes}
                    common = set.intersection(*(set(ends.tolist()) for ends, _ in aggregated.values()))
                    for size, (ends, averaged) in aggregated.items():
                        truth = y[ends]
                        keep = np.array([end in common for end in ends], dtype=bool)
                        common_y, common_prob = truth[keep], averaged[keep]
                        fold_chunks[size].append((truth, averaged, common_y, common_prob))
                        pooled[size].append((truth, averaged))
                        matched[size].append((common_y, common_prob))
                        prefix = dict(fold=fold_id, task=task, model=model, aggregation=size, **metadata[fold_id])
                        session_results.append(dict(**prefix, session=session, label=rows[session]['label'],
                            source_windows=len(y), evaluated_windows=len(truth), coverage=len(truth) / len(y),
                            common_windows=len(common_y), **metric(truth, averaged, len(names))))
                        # Preserve endpoints and probabilities so smoothing is manually auditable.
                        target = output / f'fold{fold_id}_{task}_{model}_g{size}_{list(rows).index(session):02d}_predictions.csv'
                        with target.open('w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['session', 'start_window', 'end_window', 'common_endpoint',
                                             'true_label', 'predicted_label', *['probability_' + n for n in names]])
                            for i, end in enumerate(ends):
                                writer.writerow([session, indices[end - size + 1], indices[end], bool(keep[i]),
                                                 names[truth[i]], names[averaged[i].argmax()], *averaged[i]])
                for size, chunks in fold_chunks.items():
                    y, prob, common_y, common_prob = [np.concatenate([r[i] for r in chunks]) for i in range(4)]
                    fold_results.append(dict(fold=fold_id, task=task, model=model, aggregation=size,
                        **metadata[fold_id], evaluated_windows=len(y), common_windows=len(common_y),
                        **metric(y, prob, len(names)), common_endpoint_metrics=metric(common_y, common_prob, len(names))))
            for size in sizes:
                y, prob = [np.concatenate([r[i] for r in pooled[size]]) for i in (0, 1)]
                common_y, common_prob = [np.concatenate([r[i] for r in matched[size]]) for i in (0, 1)]
                group = [r for r in fold_results if r['task'] == task and r['model'] == model and r['aggregation'] == size]
                per_session = [r for r in session_results if r['task'] == task and r['model'] == model and r['aggregation'] == size]
                scores = metric(y, prob, len(names))
                record = dict(task=task, model=model, aggregation=size, observation_seconds=size * config['window_seconds'],
                    folds=len(folds), valid_folds=sum(r['evaluated_windows'] > 0 for r in group),
                    evaluated_windows=len(y), source_windows=sum(r['source_windows'] for r in per_session),
                    common_windows=len(common_y),
                    fold_metrics=mean_metrics(group, scores),
                    common_fold_metrics=mean_metrics([r['common_endpoint_metrics'] for r in group], scores))
                if protocol == 'loso':
                    record.update(pooled=scores, common_endpoint_metrics=metric(common_y, common_prob, len(names)),
                                  session_accuracy=mean_metrics(per_session, ['accuracy'])['accuracy'])
                    class_statistics(output / f'{task}_{model}_g{size}', y, prob, names)
                else:
                    record['held_out_attack_metrics'] = {attack: mean_metrics(
                        [r for r in group if r['held_out_attack'] == attack], scores)
                        for attack in CLASSES if attack != 'normal'}
                    record['note'] = 'Equal-fold metrics; repeated attack sessions across normal rotations are not independent samples.'
                summary.append(record)
    write_csv(output / 'sessions.csv', session_results)
    write_json(output / 'folds.json', fold_results)
    write_json(output / 'config.json', dict(source=str(source), protocol=protocol, aggregations=sizes,
               comparison='common_endpoint_metrics uses exactly the same final window positions for every aggregation; threshold fixed at 0.5'))
    write_json(output / 'summary.json', summary)
    print(f'Analyzed {len(folds)} folds; {len(summary)} groups. No training. Results: {output}', flush=True)


def compare(args):
    """Collect LOSO representations/durations and explicitly paired augmentation effects."""
    table, sources = [], {}
    for directory in args.results:
        directory = Path(directory).resolve()
        config = json.loads((directory / 'config.json').read_text())
        if config.get('protocol', 'loso') != 'loso':
            raise ValueError('compare is for LOSO representation/duration/augmentation comparisons')
        dataset = json.loads(Path(config['dataset']).read_text())
        if sources:
            first_dataset = next(iter(sources.values()))[1]
            assert dataset['folds'] == first_dataset['folds'], 'Comparison requires identical session folds'
        for record in json.loads((directory / 'summary.json').read_text()):
            key = (config['window_seconds'], record['task'], record['model'], record['aggregation'], config['augmentation'])
            if key in sources:
                raise ValueError(f'Duplicate comparison configuration: {key}')
            sources[key] = (config, dataset, record)
            table.append(dict(window_seconds=key[0], task=key[1], model=key[2], aggregation=key[3],
                augmentation=key[4], effective_batch_size=config['effective_batch_size'],
                test_windows=record['test_windows'], **record['pooled'], source=str(directory)))
    # CSV columns differ between binary and multi-class tasks.
    columns = list(dict.fromkeys(k for row in table for k in row))
    output = Path(args.destination)
    effects = []
    for key, (config, dataset, record) in sources.items():
        if key[-1] != 'signal' or key[2] in ('rf', 'rf_frequency'):
            continue
        baseline_key = (*key[:-1], 'none')
        if baseline_key not in sources:
            continue
        baseline_config, baseline_dataset, baseline = sources[baseline_key]
        for field in ('split_seed', 'effective_epochs', 'effective_batch_size', 'learning_rate', 'sampling',
                      'target_rate_hz', 'checkpoint_selection', 'smoke', 'numpy_version', 'torch_version'):
            assert config[field] == baseline_config[field], f'Unmatched ablation setting: {field}'
        for left, right in zip(dataset['sessions'], baseline_dataset['sessions']):
            for field in ('session', 'candidate_windows', 'excluded_window_indices', 'source_window_samples'):
                assert left[field] == right[field], f'Unmatched ablation windows: {field}'
        assert dataset['target_window_samples'] == baseline_dataset['target_window_samples']
        effects.append(dict(window_seconds=key[0], task=key[1], model=key[2], aggregation=key[3],
            signal=record['pooled'], none=baseline['pooled'],
            delta_signal_minus_none={k: record['pooled'][k] - baseline['pooled'][k]
                                     if record['pooled'][k] is not None and baseline['pooled'][k] is not None else None
                                     for k in record['pooled']}))
    write_csv(output / 'comparison.csv', [{k: row.get(k) for k in columns} for row in table])
    write_json(output / 'augmentation_effects.json', effects)
    print(f'Collected {len(table)} configurations; {len(effects)} matched augmentation contrasts: {output}', flush=True)
