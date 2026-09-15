"""NIC electromagnetic signal baselines with strictly session-disjoint splits."""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np

DEFAULT_DATA = '/home/PublicData/qc-data/SCA/Data_new'
CLASSES = ['Hping3', 'dirsearch', 'gobuster', 'nmap port', 'nmap version', 'normal', 'sql', 'xssser']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_data(args):
    root = Path(args.data).resolve()
    files = sorted(root.rglob('*.csv'))
    if len({p.resolve() for p in files}) != len(files):
        raise ValueError('Multiple CSV paths reference the same source file')
    if set(p.parent.name for p in files) != set(CLASSES):
        raise ValueError('Unexpected class directories')
    rows = []
    for path in files:
        with path.open('rb') as f:
            header = [f.readline().decode('utf-8-sig').strip() for _ in range(3)]
            if header != ['时间,通道 B', '(s),(V)', '']:
                raise ValueError(f'Unexpected Pico header: {path}: {header}')
            start = f.tell()
            first = np.loadtxt([f.readline() for _ in range(10000)], delimiter=',')
            f.seek(start)
            count = 0
            last_byte = b''
            while block := f.read(32 * 1024 * 1024):
                count += block.count(b'\n')
                last_byte = block[-1:]
            if last_byte != b'\n':
                count += 1
            f.seek(max(start, path.stat().st_size - 1024 * 1024))
            tail = f.read().splitlines()[1:]
            last = np.loadtxt(tail[-10000:], delimiter=',')
        t0, t1 = float(first[0, 0]), float(last[-1, 0])
        dt = (t1 - t0) / (count - 1)
        if dt <= 0 or not np.isfinite(first[:, 0]).all() or not np.isfinite(last[:, 0]).all():
            raise ValueError(f'Invalid time or signal: {path}')
        # Pico prints eight decimals; sub-microsecond intervals can be quantized.
        for sample in (first, last):
            if np.max(np.abs(np.diff(sample[:, 0]) - dt)) > max(1.1e-8, dt * 1e-3):
                raise ValueError(f'Irregular endpoint timing: {path}')
        row = dict(session=str(path.relative_to(root)), label=path.parent.name,
                   bytes=path.stat().st_size, samples=count, time_start_s=t0,
                   time_end_s=t1, sample_rate_hz=1 / dt, duration_s=count * dt)
        rows.append(row)
        print(json.dumps(row), flush=True)
        write_json(Path(args.output) / 'inventory.json', dict(data_root=str(root), sessions=rows,
                   timing_check='Exact line counts; timestamp spacing checked in first/last 10000 rows. Full timing validation during preparation.'))
        write_csv(Path(args.output) / 'inventory.csv', rows)
    print(f'Inspected {len(rows)} sessions', flush=True)


def make_splits(rows, seed):
    rng = random.Random(seed)
    groups = {c: sorted(r['session'] for r in rows if r['label'] == c) for c in CLASSES}
    if any(len(v) < 2 for v in groups.values()):
        raise ValueError('At least two sessions per class required')
    for ids in groups.values():
        rng.shuffle(ids)
    folds = []
    test_order = [s for k in range(max(map(len, groups.values())))
                  for ids in groups.values() for s in ids[k:k + 1]]
    for k, held_out in enumerate(test_order):
        split = dict(train=[], val=[], test=[])
        for ids in groups.values():
            test = [held_out] if held_out in ids else []
            remaining = [s for s in ids if s not in test]
            val = [remaining[k % len(remaining)]] if len(remaining) >= 2 else []
            split['test'].extend(test)
            split['val'].extend(val)
            split['train'].extend(s for s in remaining if s not in val)
        check_split(split, rows)
        folds.append(split)
    tested = [s for split in folds for s in split['test']]
    assert len(tested) == len(set(tested)) == len(rows), 'Test coverage must be exactly once'
    return folds


def check_split(split, rows):
    a, b, c = (set(split[x]) for x in ('train', 'val', 'test'))
    assert not (a & b or a & c or b & c), 'Session leakage'
    assert a | b | c == {r['session'] for r in rows}, 'Missing sessions'
    assert all(len(split[x]) == len(set(split[x])) for x in split)



def cache_directory(args):
    out = Path(args.output)
    # Keep the original 10 ms cache intact; each new duration has its own cache.
    if args.window_seconds != 0.01:
        out = out / f'window_{args.window_seconds:g}s'
    return out / ('smoke_cache' if args.smoke else 'cache')

def prepare(args):
    import pandas as pd
    from scipy.signal import resample_poly, firwin
    from fractions import Fraction

    out = Path(args.output)
    inventory = json.loads((out / 'inventory.json').read_text())
    rows = inventory['sessions']
    root = Path(inventory['data_root'])
    if sorted(str(p.relative_to(root)) for p in root.rglob('*.csv')) != sorted(r['session'] for r in rows):
        raise ValueError('Inventory is incomplete or stale; rerun inspect')
    # Split sessions BEFORE any window extraction. Cached windows inherit only this split.
    folds = make_splits(rows, args.seed)
    cache = cache_directory(args)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'dataset.json').unlink(missing_ok=True)
    if args.target_rate > min(r['sample_rate_hz'] for r in rows):
        raise ValueError('Target rate must not exceed the lowest source rate')
    target_n = round(args.window_seconds * args.target_rate)
    if target_n < 256:
        raise ValueError('Window must contain at least 256 target samples')
    records = []
    for idx, row in enumerate(rows):
        source_n = round(args.window_seconds * row['sample_rate_hz'])
        n_windows = row['samples'] // source_n
        used = min(args.smoke_windows, n_windows) if args.smoke else n_windows
        if used < 1:
            raise ValueError(f'Session too short: {row["session"]}')
        path = root / row['session']
        if path.stat().st_size != row['bytes']:
            raise ValueError(f'Source size changed: {path}')
        destination = cache / f'{idx:02d}.npy'
        signal = np.lib.format.open_memmap(destination, mode='w+', dtype='float32', shape=(used, target_n))
        ratio = Fraction(target_n, source_n)
        # Match scipy's default filter exactly, but build it only once per session.
        # Long windows at fractional source rates can require millions of FIR taps.
        max_rate = max(ratio.numerator, ratio.denominator)
        fir = (firwin(20 * max_rate + 1, 1. / max_rate, window=('kaiser', 5.0))
               if max_rate > 1 else ('kaiser', 5.0))
        offset = 0
        written = 0
        excluded = []
        invalid_samples = 0
        # Chunk boundaries coincide with windows; the final partial window is discarded.
        chunk_size = source_n * max(1, 1000000 // source_n)
        reader = pd.read_csv(path, skiprows=3, header=None, names=['time', 'signal'],
                             dtype=np.float64, chunksize=chunk_size,
                             nrows=used * source_n if args.smoke else None)
        with reader:
            for chunk in reader:
                values = chunk.to_numpy()
                bad_time = np.flatnonzero(~np.isfinite(values[:, 0]))
                if len(bad_time):
                    raise ValueError(f'Nonfinite timestamp: {path}, CSV line {offset + int(bad_time[0]) + 4}')
                invalid_samples += int((~np.isfinite(values[:, 1])).sum())
                expected = row['time_start_s'] + np.arange(offset, offset + len(values)) / row['sample_rate_hz']
                if np.max(np.abs(values[:, 0] - expected)) > max(2e-8, 0.02 / row['sample_rate_hz']):
                    raise ValueError(f'Timestamp gap or irregular rate: {path}, row {offset}')
                offset += len(values)
                count = min(len(values) // source_n, used - written)
                if count:
                    raw = values[:count * source_n, 1].reshape(count, source_n)
                    # Anti-aliasing resampling; common bandwidth/rate for every session and model.
                    valid = np.isfinite(raw).all(axis=1)
                    excluded.extend((written + np.flatnonzero(~valid)).tolist())
                    # Keep original window positions; invalid slots are never model inputs.
                    block = np.full((count, target_n), np.nan, dtype=np.float32)
                    if valid.any():
                        transformed = resample_poly(raw[valid], ratio.numerator, ratio.denominator, axis=1, window=fir)
                        assert transformed.shape == (int(valid.sum()), target_n)
                        block[valid] = transformed.astype(np.float32)
                        if not np.isfinite(block[valid]).all():
                            raise ValueError(f'Nonfinite resampling output: {path}, window {written}')
                    signal[written:written + count] = block
                    written += count
        assert written == used
        assert offset == (used * source_n if args.smoke else row['samples'])
        signal.flush()
        del signal
        if len(excluded) == used:
            raise ValueError(f'No valid complete windows remain: {path}')
        records.append(dict(**row, source_window_samples=source_n, available_windows=n_windows,
                            candidate_windows=used, windows=used - len(excluded),
                            invalid_signal_samples=invalid_samples, excluded_windows=len(excluded),
                            excluded_window_indices=excluded, dropped_tail_samples=row['samples'] % source_n,
                            cache_file=str(destination.resolve())))
        print(f'Prepared {row["session"]}: {used - len(excluded)}/{n_windows} valid windows; '
              f'excluded={len(excluded)}, nonfinite_signal_samples={invalid_samples}', flush=True)
    config = dict(seed=args.seed, smoke=args.smoke, window_seconds=args.window_seconds,
                  target_rate_hz=args.target_rate, target_window_samples=target_n,
                  classes=CLASSES, sessions=records, folds=folds)
    write_json(cache / 'dataset.json', config)
    audit = []
    for k, split in enumerate(folds):
        check_split(split, rows)
        for part, ids in split.items():
            for row in records:
                if row['session'] in ids:
                    audit.append(dict(fold=k, split=part, **row))
    write_csv(cache / 'session_splits.csv', audit)
    print(f'Session leakage checks passed for {len(folds)} folds', flush=True)


def statistics(x):
    x = x.astype(np.float64)
    energy = np.square(x).sum(axis=1)
    return np.column_stack([np.sqrt(energy / x.shape[1]), energy, x.std(axis=1),
                            np.ptp(x, axis=1), x.mean(axis=1),
                            np.abs(x).mean(axis=1), np.abs(x).max(axis=1)]).astype(np.float32)


def load_signal_tensor(arrays, indices, device):
    """Load valid windows once, preserving session and original-window ordering."""
    import torch

    count = sum(len(ids) for ids in indices.values())
    width = next(iter(arrays.values())).shape[1]
    signal = torch.empty((count, 1, width), dtype=torch.float32, device=device)
    session_ids = {}
    offset = 0
    for session, array in arrays.items():
        valid = indices[session]
        session_ids[session] = torch.arange(offset, offset + len(valid), device=device)
        chunk = max(1, 2_000_000 // width)
        for start in range(0, len(valid), chunk):
            block = array[valid[start:start + chunk]]
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite values in valid cached windows: {session}')
            signal[offset + start:offset + start + len(block), 0].copy_(torch.from_numpy(block))
        offset += len(valid)
    return signal, session_ids


def iter_batches(signal, labels, ids, batch_size, generator=None):
    """Shuffle only the selected split and gather whole batches on its device."""
    import torch

    order = ids if generator is None else ids[torch.randperm(len(ids), device=ids.device, generator=generator)]
    for start in range(0, len(order), batch_size):
        selected = order[start:start + batch_size]
        yield signal[selected], labels[selected]



STAT_NAMES = ['rms', 'energy', 'std', 'peak_to_peak', 'mean', 'mean_abs', 'max_abs']


def frequency_statistics(x, sample_rate):
    """Welch AC power density: fixed relative-Nyquist bands, no fitted transforms."""
    from scipy.signal import welch
    freq, psd = welch(x, fs=sample_rate, nperseg=min(4096, x.shape[1]),
                      detrend='constant', axis=1)
    df = freq[1] - freq[0]
    power = psd.astype(np.float64) * df
    total = power.sum(axis=1)
    denom = np.maximum(total, np.finfo(np.float64).tiny)
    edges = np.array([0, .02, .1, .2, .4, .7, 1.]) * sample_rate / 2
    columns, names = [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (freq >= lo) & ((freq <= hi) if i == len(edges) - 2 else (freq < hi))
        band = power[:, mask].sum(axis=1)
        columns.extend([band * x.shape[1] / sample_rate, band / denom])
        names.extend([f'band_{lo:g}_{hi:g}_Hz_energy', f'band_{lo:g}_{hi:g}_Hz_ratio'])
    q = power / denom[:, None]
    peak = freq[np.argmax(power, axis=1)]
    peak[total == 0] = 0
    columns.extend([total, peak, (power * freq).sum(axis=1) / denom,
                    -(q * np.log(np.maximum(q, np.finfo(float).tiny))).sum(axis=1) / np.log(len(freq))])
    names.extend(['ac_power', 'dominant_frequency_hz', 'spectral_centroid_hz', 'spectral_entropy'])
    return np.column_stack(columns).astype(np.float32), names


def balanced_ids(split, session_ids, label_map, binary, seed):
    """Exact class quotas, then near-equal session quotas, replacement within session."""
    rng = np.random.default_rng(seed)
    size = sum(len(session_ids[s]) for s in split['train'])
    # Binary: seven parts normal and one part per attack type (14 parts total).
    unit = 14 if binary else len(CLASSES)
    size = max(unit, ((size + unit - 1) // unit) * unit)
    drawn, audit = [], []
    for c, name in enumerate(CLASSES):
        sessions = [s for s in split['train'] if label_map[s] == c]
        quota = size // unit * (7 if binary and name == 'normal' else 1)
        shuffled = rng.permutation(sessions).tolist()
        for i, session in enumerate(shuffled):
            count = quota // len(sessions) + int(i < quota % len(sessions))
            ids = session_ids[session]
            drawn.append(rng.choice(ids, size=count, replace=True))
            audit.append(dict(session=session, label=name, available_windows=len(ids), draws=count))
    result = np.concatenate(drawn)
    rng.shuffle(result)
    assert set(result).issubset(set(np.concatenate([session_ids[s] for s in split['train']])))
    return result, audit


def augment_signal(x, generator):
    """Train only; offsets/noise relative to each input window's standard deviation."""
    import torch
    scale = x.std(dim=-1, correction=0, keepdim=True)
    shape = (len(x), 1, 1)
    amplitude = .8 + .4 * torch.rand(shape, device=x.device, generator=generator)
    offset = (.2 * torch.rand(shape, device=x.device, generator=generator) - .1) * scale
    noise = torch.randn(x.shape, device=x.device, generator=generator) * (.01 * scale)
    return amplitude * x + offset + noise


def aggregate_probabilities(indices, probabilities, size):
    """Stride-one trailing averages; only complete runs of original adjacent windows."""
    if size <= 0 or len(indices) != len(probabilities):
        raise ValueError('Invalid aggregation inputs')
    assert np.all(np.diff(indices) > 0), 'Window indices must be strictly increasing'
    ends = np.arange(size - 1, len(indices), dtype=np.int64)
    ends = ends[indices[ends] - indices[ends - size + 1] == size - 1]
    cumulative = np.vstack([np.zeros((1, probabilities.shape[1])),
                            np.cumsum(probabilities, axis=0, dtype=np.float64)])
    averaged = (cumulative[ends + 1] - cumulative[ends - size + 1]) / size
    return ends, averaged


def metric(y, probabilities, n):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
    keys = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    if n == 2:
        keys += ['auroc', 'auprc', 'balanced_accuracy', 'normal_recall', 'attack_recall']
    if len(y) == 0:
        return dict.fromkeys(keys)
    pred = probabilities.argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(n)),
                                                average='macro', zero_division=0)
    scores = dict(accuracy=float(accuracy_score(y, pred)), macro_precision=float(p),
                  macro_recall=float(r), macro_f1=float(f))
    if n == 2:
        recalls = [float((pred[y == c] == c).mean()) if np.any(y == c) else None for c in (0, 1)]
        both = all(v is not None for v in recalls)
        scores.update(auroc=float(roc_auc_score(y, probabilities[:, 1])) if both else None,
                      auprc=float(average_precision_score(y, probabilities[:, 1])) if both else None,
                      balanced_accuracy=float(np.mean(recalls)) if both else None,
                      normal_recall=recalls[0], attack_recall=recalls[1])
    return scores

def save_cm(path, y, pred, names):
    from sklearn.metrics import confusion_matrix
    cm = (confusion_matrix(y, pred, labels=list(range(len(names)))) if len(y)
          else np.zeros((len(names), len(names)), dtype=np.int64))
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true / predicted'] + names)
        writer.writerows([[name] + row.tolist() for name, row in zip(names, cm)])


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch import nn
    from sklearn.ensemble import RandomForestClassifier
    cache = cache_directory(args)
    config = json.loads((cache / 'dataset.json').read_text())
    assert config['smoke'] == args.smoke
    assert config['window_seconds'] == args.window_seconds, 'Wrong window cache'
    rows = config['sessions']
    result_dir = cache.parent / (('smoke_results' if args.smoke else 'results') +
                                 f'_{args.sampling}_{args.augmentation}')
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / 'summary.json').unlink(missing_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    assert len({Path(r['cache_file']).resolve() for r in rows}) == len(rows), 'Shared session cache files'
    arrays = {r['session']: np.load(r['cache_file'], mmap_mode='r') for r in rows}
    indices = {}
    for r in rows:
        assert arrays[r['session']].shape == (r['candidate_windows'], config['target_window_samples'])
        keep = np.ones(r['candidate_windows'], dtype=bool)
        excluded = np.asarray(r['excluded_window_indices'], dtype=np.int64)
        assert len(excluded) == len(set(excluded.tolist())) == r['excluded_windows']
        assert ((excluded >= 0) & (excluded < len(keep))).all()
        keep[excluded] = False
        indices[r['session']] = np.flatnonzero(keep)
        assert len(indices[r['session']]) == r['windows'] > 0
    label_map = {r['session']: CLASSES.index(r['label']) for r in rows}
    chunk_size = max(1, 2_000_000 // config['target_window_samples'])
    features = {}
    frequency_names = None
    for session, array in arrays.items():
        blocks = []
        for start in range(0, len(indices[session]), chunk_size):
            x = array[indices[session][start:start + chunk_size]]
            frequency, frequency_names = frequency_statistics(x, config['target_rate_hz'])
            blocks.append(np.column_stack([statistics(x), frequency]))
        features[session] = np.concatenate(blocks)
    all_features = np.concatenate(list(features.values()))
    # Cap activations for 0.5--5 s inputs without altering the actual input signal.
    batch_size = min(args.batch_size, max(1, args.max_batch_points // config['target_window_samples']))
    started = time.perf_counter()
    signal, session_ids = load_signal_tensor(arrays, indices, device)
    all_labels = torch.cat([torch.full((len(indices[s]),), label_map[s], dtype=torch.long, device=device)
                            for s in arrays])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(f'Effective batch size: {batch_size}; input samples/window: {config["target_window_samples"]}', flush=True)
    print(f'Loaded {len(signal)} valid windows onto {device}: '
          f'{signal.numel() * signal.element_size() / 2**30:.2f} GiB in {time.perf_counter() - started:.2f}s', flush=True)

    class CNN(nn.Module):
        def __init__(self, mode, n_classes):
            super().__init__()
            self.mode = mode
            if mode == 'stft_cnn':
                self.register_buffer('window', torch.hann_window(128))
                self.body = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
                self.head = nn.Linear(32 * 32, n_classes)
            else:
                self.body = nn.Sequential(nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.ReLU(),
                    nn.Conv1d(16, 32, 7, stride=4, padding=3), nn.ReLU(),
                    nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU())
                self.head = nn.Linear(64, n_classes)

        def forward(self, x):
            if self.mode == 'zscore_cnn':
                x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, correction=0, keepdim=True).clamp_min(1e-8)
            elif self.mode == 'stft_cnn':
                z = torch.stft(x[:, 0], n_fft=128, hop_length=32, window=self.window,
                               center=False, return_complex=True)
                x = torch.log1p(z.abs().square())[:, None]
            # Average over time; STFT retains its frequency axis.
            return self.head(self.body(x).mean(dim=-1).flatten(1))

    def labels(ids, binary):
        return np.concatenate([np.full(len(indices[s]), int(label_map[s] != CLASSES.index('normal'))
                               if binary else label_map[s], dtype=np.int64) for s in ids])

    epochs = 1 if args.smoke else args.epochs
    settings = dict(vars(args), split_seed=config['seed'], effective_epochs=epochs, effective_batch_size=batch_size,
                    window_seconds=config['window_seconds'], target_rate_hz=config['target_rate_hz'], dataset=str((cache / 'dataset.json').resolve()),
                    numpy_version=np.__version__, torch_version=torch.__version__,
                    sklearn_version=__import__('sklearn').__version__,
                    scipy_version=__import__('scipy').__version__,
                    pandas_version=__import__('pandas').__version__,
                    metrics='Fixed 2/8 macro labels; binary AUROC/AUPRC/balanced accuracy require both labels. Missing class recall and empty aggregation = null. Pooled results combine held-out predictions from different fold models.',
                    feature_names=STAT_NAMES, frequency_feature_names=STAT_NAMES + frequency_names,
                    aggregation='Trailing stride-one probability mean, within original contiguous session windows only',
                    augmentation_parameters=dict(amplitude=[.8, 1.2], offset_std=[-.1, .1], noise_std=.01))
    write_json(result_dir / 'config.json', settings)
    results = []
    sampling_audit = []
    cpu_session_ids = {s: ids.cpu().numpy() for s, ids in session_ids.items()}
    pooled = {}
    assert config['folds'] == make_splits(rows, config['seed']), 'Stored folds changed'
    for fold_id, split in enumerate(config['folds']):
        check_split(split, rows)
        batch_ids = {part: torch.cat([session_ids[s] for s in ids]) for part, ids in split.items()}
        for task in ('binary', 'multiclass'):
            binary = task == 'binary'
            targets = (all_labels != CLASSES.index('normal')).long() if binary else all_labels
            names = ['normal', 'attack'] if binary else CLASSES
            n = len(names)
            ys = {part: labels(ids, binary) for part, ids in split.items()}
            assert set(ys['train']) == set(range(n)), 'Training class missing'
            epoch_ids = []
            for epoch in range(epochs):
                if args.sampling == 'balanced':
                    selected, audit = balanced_ids(split, cpu_session_ids, label_map, binary,
                                                   config['seed'] + fold_id + epoch * 1000)
                else:
                    selected = batch_ids['train'].cpu().numpy().copy()
                    np.random.default_rng(config['seed'] + fold_id + epoch * 1000).shuffle(selected)
                    audit = [dict(session=s, label=CLASSES[label_map[s]],
                                  available_windows=len(cpu_session_ids[s]), draws=len(cpu_session_ids[s]))
                             for s in split['train']]
                epoch_ids.append(torch.as_tensor(selected, device=device))
                sampling_audit.extend(dict(fold=fold_id, task=task, epoch=epoch + 1, **row) for row in audit)
            write_csv(result_dir / 'training_sampling.csv', sampling_audit)
            for mode in ('rf', 'rf_frequency', 'raw_cnn', 'zscore_cnn', 'stft_cnn'):
                seed = config['seed'] + fold_id
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                prefix = result_dir / f'fold{fold_id}_{task}_{mode}'
                history = []
                if mode in ('rf', 'rf_frequency'):
                    model = RandomForestClassifier(n_estimators=10 if args.smoke else 100,
                        class_weight='balanced' if args.sampling == 'window' else None,
                        random_state=seed, n_jobs=args.threads)
                    selected = epoch_ids[0].cpu().numpy()
                    width = len(STAT_NAMES) if mode == 'rf' else all_features.shape[1]
                    model.fit(all_features[selected, :width], targets[epoch_ids[0]].cpu().numpy())
                    probabilities = model.predict_proba(all_features[batch_ids['test'].cpu().numpy(), :width])
                    val_prob = model.predict_proba(all_features[batch_ids['val'].cpu().numpy(), :width])
                    val_scores = metric(ys['val'], val_prob, n)
                    import joblib
                    joblib.dump(model, str(prefix) + '.joblib')
                    best_epoch = None
                else:
                    generator = torch.Generator(device=device).manual_seed(seed)
                    model = CNN(mode, n).to(device)
                    counts = np.bincount(ys['train'], minlength=n)
                    weights = (torch.tensor(len(ys['train']) / (n * counts), dtype=torch.float32, device=device)
                               if args.sampling == 'window' else None)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                    best_loss = float('inf')
                    for epoch in range(epochs):
                        epoch_started = time.perf_counter()
                        model.train()
                        total_loss = torch.zeros((), device=device)
                        for x, y in iter_batches(signal, targets, epoch_ids[epoch], batch_size):
                            if args.augmentation == 'signal':
                                x = augment_signal(x, generator)
                            optimizer.zero_grad()
                            loss = nn.functional.cross_entropy(model(x), y, weight=weights)
                            loss.backward()
                            optimizer.step()
                            total_loss.add_(loss.detach() * len(y))
                        model.eval()
                        val_loss = torch.zeros((), device=device)
                        with torch.no_grad():
                            for x, y in iter_batches(signal, targets, batch_ids['val'], batch_size):
                                val_loss.add_(nn.functional.cross_entropy(model(x), y, reduction='sum'))
                        train_loss = total_loss.item() / len(epoch_ids[epoch])
                        val_loss = val_loss.item() / len(ys['val'])
                        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                            raise FloatingPointError(f'Nonfinite epoch loss: fold {fold_id}, {task}, {mode}')
                        epoch_seconds = time.perf_counter() - epoch_started
                        history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss, epoch_seconds=epoch_seconds))
                        if val_loss < best_loss:
                            best_loss, best_epoch = val_loss, epoch + 1
                            torch.save(model.state_dict(), str(prefix) + '.pt')
                        print(f'fold={fold_id} {task} {mode} epoch={epoch + 1} val_loss={val_loss:.5f} seconds={epoch_seconds:.2f}', flush=True)
                    model.load_state_dict(torch.load(str(prefix) + '.pt', map_location=device, weights_only=True))
                    def predict(part):
                        with torch.no_grad():
                            return torch.cat([model(x).softmax(1) for x, _ in
                                              iter_batches(signal, targets, batch_ids[part], batch_size)]).cpu().numpy()
                    probabilities = predict('test')
                    val_scores = metric(ys['val'], predict('val'), n)
                    write_json(Path(str(prefix) + '_history.json'), history)
                for aggregation in (1, 3, 5, 10):
                    truth_chunks, probability_chunks, predictions = [], [], []
                    pos = 0
                    for session in split['test']:
                        size = len(indices[session])
                        ends, averaged = aggregate_probabilities(indices[session], probabilities[pos:pos + size], aggregation)
                        truth = ys['test'][pos:pos + size][ends]
                        truth_chunks.append(truth)
                        probability_chunks.append(averaged)
                        for j, end in enumerate(ends):
                            predictions.append(dict(session=session,
                                start_window=int(indices[session][end - aggregation + 1]),
                                end_window=int(indices[session][end]), true_label=names[truth[j]],
                                predicted_label=names[averaged[j].argmax()],
                                **{f'probability_{name}': float(averaged[j, c]) for c, name in enumerate(names)}))
                        pos += size
                    truth = np.concatenate(truth_chunks)
                    prob = np.concatenate(probability_chunks)
                    scores = dict(fold=fold_id, task=task, model=mode, aggregation=aggregation,
                                  test_windows=len(truth), source_test_windows=len(ys['test']),
                                  coverage=len(truth) / len(ys['test']),
                                  unavailable_reason=None if len(truth) else 'No complete contiguous run',
                                  test_sessions=split['test'], best_epoch=best_epoch,
                                  **metric(truth, prob, n), validation=val_scores)
                    results.append(scores)
                    write_json(result_dir / 'metrics.json', results)
                    prediction_path = Path(str(prefix) + f'_aggregate{aggregation}_predictions.csv')
                    if predictions:
                        write_csv(prediction_path, predictions)
                    else:
                        prediction_path.write_text('session,start_window,end_window,true_label,predicted_label,' +
                                                   ','.join(f'probability_{name}' for name in names) + '\n')
                    if not binary:
                        save_cm(Path(str(prefix) + f'_aggregate{aggregation}_confusion.csv'), truth, prob.argmax(axis=1), names)
                    pooled.setdefault((task, mode, aggregation), []).append((truth, prob))
                print(f'fold={fold_id} {task} {mode} accuracy={results[-4]["accuracy"]}', flush=True)
    summary = []
    for (task, mode, aggregation), chunks in pooled.items():
        group = [r for r in results if r['task'] == task and r['model'] == mode and r['aggregation'] == aggregation]
        truth, prob = (np.concatenate([pair[i] for pair in chunks]) for i in (0, 1))
        pooled_scores = metric(truth, prob, 2 if task == 'binary' else 8)
        means = {}
        for key in pooled_scores:
            values = [r[key] for r in group if r[key] is not None]
            means[key] = dict(mean=float(np.mean(values)) if values else None,
                              std=float(np.std(values)) if values else None, valid_folds=len(values))
        summary.append(dict(task=task, model=mode, aggregation=aggregation, folds=len(group),
                            valid_folds=sum(r['test_windows'] > 0 for r in group), test_windows=len(truth),
                            source_test_windows=sum(r['source_test_windows'] for r in group),
                            fold_metrics=means, pooled=pooled_scores))
        if task == 'multiclass':
            save_cm(result_dir / f'{task}_{mode}_aggregate{aggregation}_pooled_confusion.csv',
                    truth, prob.argmax(axis=1), CLASSES)
    write_json(result_dir / 'summary.json', summary)
    print(f'Completed {len(results)} evaluations; all session leakage assertions passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect_parser = commands.add_parser('inspect', help='Count CSV samples and inspect format/timing')
    inspect_parser.add_argument('--data', default=DEFAULT_DATA)
    prepare_parser = commands.add_parser('prepare', help='Split sessions then extract windows')
    prepare_parser.add_argument('--smoke-windows', type=int, default=4)
    prepare_parser.add_argument('--seed', type=int, default=42)
    prepare_parser.add_argument('--window-seconds', type=float, default=0.01)
    prepare_parser.add_argument('--target-rate', type=int, default=100000)
    run_parser = commands.add_parser('run', help='Evaluate five models on both tasks')
    run_parser.add_argument('--window-seconds', type=float, default=0.01)
    run_parser.add_argument('--sampling', choices=['balanced', 'window'], default='balanced')
    run_parser.add_argument('--augmentation', choices=['none', 'signal'], default='signal')
    run_parser.add_argument('--max-batch-points', type=int, default=1024000)
    run_parser.add_argument('--epochs', type=int, default=20)
    run_parser.add_argument('--batch-size', type=int, default=1024)
    run_parser.add_argument('--learning-rate', type=float, default=1e-3)
    run_parser.add_argument('--threads', type=int, default=4)
    run_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    for command in (inspect_parser, prepare_parser, run_parser):
        command.add_argument('--output', default='outputs')
    for command in (prepare_parser, run_parser):
        command.add_argument('--smoke', action='store_true', help='Bounded session prefixes (prepare --smoke-windows), one epoch, 10 RF trees')
    args = parser.parse_args()
    for name in ('window_seconds', 'target_rate', 'epochs', 'batch_size', 'learning_rate', 'threads', 'max_batch_points', 'smoke_windows'):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    {'inspect': inspect_data, 'prepare': prepare, 'run': run}[args.command](args)


if __name__ == '__main__':
    main()
