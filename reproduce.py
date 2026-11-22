"""Re-evaluate the manuscript E01--E10 release. No model or network is used."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'release_manifest.json').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'code/vendor'))
DATA_ROOT = Path(os.environ.get('TAR_DATA_DIR', str(ROOT / 'data'))).expanduser().resolve()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def data_file(relative, data_dir=None):
    path = (data_dir or DATA_ROOT) / relative
    if not path.is_file():
        raise FileNotFoundError(
            f'Dataset input is not distributed with this repository: {path}. '
            'Obtain authorized data from the official source; see MODELS.md and --data-dir.')
    manifest = ROOT / 'evaluation_inputs.json'
    if manifest.exists():
        expected = load(manifest).get(relative)
        if expected and (path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']):
            raise ValueError(f'External evaluation input differs from the frozen study: {relative}')
    return path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def offline():
    original = socket.socket.connect
    def blocked(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise PermissionError('Release reproduction is offline; no IP connections are allowed')
        return original(sock, address)
    socket.socket.connect = blocked


def score(run, task, split, scope, cache):
    from recare_baselines.ranked_eval import evaluate_ranked_files, read_qrels, read_ranked
    path = ROOT / run['path']
    if sha(path) != run['sha256']:
        raise ValueError(f'Ranking checksum differs: {run["path"]}')
    key = (run['sha256'], task, split, tuple(scope or []))
    if key in cache:
        return cache[key]
    gold_path = data_file(f'qrels-{task}/{split}.jsonl')
    rankings = read_ranked(path)
    if scope:
        if len(scope) != len(set(scope)) or set(rankings) != set(scope):
            raise ValueError('Frozen subset query membership changed')
        gold = read_qrels(gold_path)
        if not set(scope) <= set(gold):
            raise ValueError('Frozen scope contains a query outside the released split')
        with tempfile.TemporaryDirectory(prefix='tar_scope_') as folder:
            subset = Path(folder) / 'qrels.jsonl'
            with subset.open('w', encoding='utf-8') as stream:
                for q in scope:
                    for d, g in gold[q].items():
                        stream.write(json.dumps({'query-id': q, 'corpus-id': d, 'score': g}) + '\n')
            result = evaluate_ranked_files(path, subset, filter_self=task.startswith('rev'), max_docs=200)
    else:
        result = evaluate_ranked_files(path, gold_path, filter_self=task.startswith('rev'), max_docs=200)
    if (result.missing_qids or result.extra_qids or result.n_duplicate_docs_dropped
            or result.n_self_matches_dropped or result.n_truncated_queries):
        raise ValueError(f'Invalid query coverage or candidate list: {run["path"]}')
    cache[key] = asdict(result)
    return cache[key]


def verify_experiment(experiment, cache):
    data = load(ROOT / experiment / 'results.json')
    rows, by_entry = [], {}
    for entry in data:
        runs = entry['rankings']
        scored = [score(r, entry['task'], entry['split'], entry.get('scope_qids'), cache) for r in runs]
        if scored:
            metrics = {m: sum(r['metrics'][m] for r in scored) / len(scored)
                       for m in entry['metrics']}
            for m, expected in entry['metrics'].items():
                if abs(metrics[m] - expected) > 1e-12:
                    raise ValueError(f'Metric mismatch: {experiment}/{entry["row_id"]}/{entry["task"]}/{m}')
            verification = 'independently_recomputed'
        else:
            if entry['source_type'] != 'official_reported':
                raise ValueError('Unclassified aggregate-only result')
            evidence = load(ROOT / entry['reported_metrics'])
            metrics = entry['metrics']
            if any(abs(evidence['metrics'][m] - v) > 1e-12 for m, v in metrics.items()):
                raise ValueError('Reported-source metric changed')
            verification = 'reported_source_checked_not_locally_reproduced'
        by_entry[entry['row_id'] + '/' + entry['task']] = scored
        rows.append({'experiment': experiment, 'row_id': entry['row_id'],
                     'configuration': entry['configuration'], 'task': entry['task'],
                     'split': entry['split'], 'source_type': entry['source_type'],
                     'verification': verification, 'repeats': len(runs),
                     'ndcg_10': metrics['ndcg_10'], 'recall_100': metrics['recall_100'],
                     'map': metrics['map'],
                     'n_queries': scored[0]['n_queries'] if scored else evidence['n_queries']})
    csv_save(ROOT / 'outputs' / experiment / 'metrics.csv', rows)
    save(ROOT / 'outputs' / experiment / 'verification.json', {
        'experiment': experiment, 'settings_checked': len(rows),
        'ranked_settings': sum(bool(e['rankings']) for e in data),
        'official_reported_settings': sum(not e['rankings'] for e in data),
        'new_inference': False, 'network_allowed': False,
    })
    return data, by_entry


def check_manifest():
    files = load(ROOT / 'file_manifest.json')
    for name, info in files.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError(f'Package file changed/missing: {name}')
    print(f'CHECKSUM_OK {len(files)} files', flush=True)


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['all'] + [f'E{i:02d}' for i in range(1, 11)], default='all')
    parser.add_argument('--check', action='store_true', help='Verify every packaged file checksum')
    parser.add_argument('--plot', action='store_true', help='Regenerate the selected table or composite figure')
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT,
                        help='Authorized local evaluation inputs; not included in the repository')
    args = parser.parse_args()
    DATA_ROOT = args.data_dir.expanduser().resolve()
    offline()
    if args.check:
        check_manifest()
    experiments = [f'E{i:02d}' for i in range(1, 11)] if args.experiment == 'all' else [args.experiment]
    cache, receipt = {}, {}
    for exp in experiments:
        data, per_query = verify_experiment(exp, cache)
        if args.plot:
            sys.path.insert(0, str(ROOT / 'code'))
            from plot_results import render
            render(ROOT, exp, data, per_query, data_dir=DATA_ROOT)
        receipt[exp] = load(ROOT / 'outputs' / exp / 'verification.json')
        print(f'{exp}: {len(data)} settings checked; no new inference', flush=True)
    save(ROOT / 'outputs/verification_summary.json', {
        'experiments': receipt, 'unique_ranked_evaluations': len(cache),
        'external_model_requests': 0, 'manuscript_unchanged': True})


if __name__ == '__main__':
    main()
