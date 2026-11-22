"""Regenerate one table OR one composite figure per manuscript experiment."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

TASKS = ('rat2rev-en', 'rev2rev-en', 'rat2rev-ja', 'rev2rev-ja')
SHORT = ('Rat-en', 'Rev-en', 'Rat-ja', 'Rev-ja')
METRICS = ('ndcg_10', 'recall_100', 'map')
NAMES = ('nDCG@10', 'R@100', 'MAP')
COLORS = ('#2466A8', '#168C83', '#D88727', '#8756A5')


def interval(values, groups):
    buckets = {}
    for q, value in values.items():
        buckets.setdefault(groups[q], []).append(value)
    sums = np.array([sum(v) for v in buckets.values()])
    sizes = np.array([len(v) for v in buckets.values()])
    draws = np.random.default_rng(20261030).integers(0, len(sums), size=(10000, len(sums)))
    return np.quantile(sums[draws].sum(axis=1) / sizes[draws].sum(axis=1), [.025, .975])


def render(root, exp, entries, per_query, data_dir=None):
    out = root / 'outputs' / exp
    rows = json.loads((root / exp / 'configs/rows.json').read_text(encoding='utf-8'))
    ids = [r['row_id'] for r in rows]
    index = {(e['row_id'], e['task']): e for e in entries}
    data = np.array([[[100 * index[r, t]['metrics'][m] if (r, t) in index else np.nan
                       for m in METRICS] for t in TASKS] for r in ids])
    if exp in ('E01', 'E02', 'E07'):
        # Generate from saved metrics; no manuscript fragments are distributed.
        replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                        '_': r'\_', '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}',
                        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
        escape = lambda value: ''.join(replacements.get(c, c) for c in str(value))
        flat = data.reshape(len(ids), -1)
        best = [int(np.nanargmax(flat[:, j])) if np.isfinite(flat[:, j]).any() else -1
                for j in range(flat.shape[1])]
        lines = [r'\begin{tabular}{l' + 'r' * 12 + '}', r'\toprule',
                 'Configuration & ' + ' & '.join(escape(t + ' ' + m) for t in SHORT for m in NAMES) + r' \\',
                 r'\midrule']
        for i, row in enumerate(rows):
            values = []
            for j, value in enumerate(flat[i]):
                cell = '--' if not np.isfinite(value) else f'{value:.2f}'
                values.append(r'\textbf{' + cell + '}' if i == best[j] else cell)
            lines.append(escape(row['configuration']) + ' & ' + ' & '.join(values) + r' \\')
        lines.extend([r'\bottomrule', r'\end{tabular}'])
        (out / 'table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    legend_axes = None
    if exp == 'E03':
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 3.25), layout='constrained')
        labels = ('Orig', 'Q2E', 'Q2D', 'TQ2E', 'G1', 'G3', 'G5', 'A1', 'A3', 'A5', 'Rep5', 'Long')
        for j, ax in enumerate(axes):
            for t, color in enumerate(COLORS):
                ax.scatter(np.arange(12) + (t - 1.5) * .12, data[:, t, j], color=color,
                           marker=('o', 's', '^', 'D')[t], s=15, label=SHORT[t])
            ax.set_xticks(range(12), labels, rotation=65, fontsize=6.8)
            ax.set_ylabel(NAMES[j] + ' x100'); ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.18)
        legend_axes = axes[0]
    elif exp == 'E04':
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 3.9), sharey=True, layout='constrained')
        output = []
        for j, (metric, ax) in enumerate(zip(METRICS, axes)):
            name = ('nDCG@10', 'R@100', 'AP')[j]
            for t, task in enumerate(TASKS):
                from reproduce import data_file
                groups = json.loads(data_file(f'event_groups/{task}.test.json', data_dir).read_text(encoding='utf-8'))['groups']
                for c, before in enumerate((ids[0], ids[1])):
                    a, b = (per_query[r + '/' + task][0]['per_query'] for r in (before, ids[2]))
                    diff = {q: b[q][name] - a[q][name] for q in a}
                    low, high = interval(diff, groups); mean = np.mean(list(diff.values()))
                    y = 7 - (2 * t + c)
                    ax.hlines(y, 100 * low, 100 * high, color=COLORS[t], linewidth=1.35)
                    ax.plot(100 * mean, y, 'o', color=COLORS[t], markersize=3.5)
                    output.append({'task': task, 'comparison': f'T2 - T{c}', 'metric': metric,
                                   'delta': mean, 'ci_low': low, 'ci_high': high})
            labels = [f'{s}  T2 - T{c}' for s in SHORT for c in (0, 1)]
            ax.set_yticks(range(8), list(reversed(labels)), fontsize=7)
            ax.tick_params(labelleft=j == 0); ax.axvline(0, color='#555', ls='--', lw=.8)
            ax.set_xlabel('Delta ' + NAMES[j] + ' (pp)'); ax.grid(axis='x', alpha=.17)
            if j == 1: ax.set_xlim(-.5, .5)
        from reproduce import csv_save
        csv_save(out / 'paired_contrasts_recomputed.csv', output)
    elif exp == 'E05':
        fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.9), layout='constrained')
        for factor, xs in enumerate(((100, 300, 500, 1000), (100, 200, 300, 500))):
            for j, ax in enumerate(axes[factor]):
                for t, color in enumerate(COLORS):
                    ax.plot(xs, data[factor * 4:factor * 4 + 4, t, j], 'o-', ms=3, lw=1.1,
                            color=color, label=SHORT[t])
                ax.set_xticks(xs); ax.tick_params(labelsize=7)
                ax.set_xlabel('Source depth S' if factor == 0 else 'Cross-encoder depth C')
                ax.set_ylabel(NAMES[j] + ' x100'); ax.grid(alpha=.16)
        legend_axes = axes.flat[0]
    elif exp == 'E06':
        fig, axes = plt.subplots(4, 3, figsize=(6.4, 7.1), layout='constrained')
        for j in range(3):
            deltas = [data[:, t, j] - data[8 if t in (0, 2) else 7, t, j] for t in range(4)]
            scale = max(max(abs(d).max() for d in deltas), .01)
            for t, ax in enumerate(axes[:, j]):
                absolute, delta = data[:, t, j].reshape(3, 3), deltas[t].reshape(3, 3)
                image = ax.imshow(delta, cmap='RdBu_r', norm=TwoSlopeNorm(0, -scale, scale), aspect='auto')
                for r in range(3):
                    for c in range(3):
                        ax.text(c, r, f'{absolute[r,c]:.2f}', ha='center', va='center', fontsize=7.6,
                                color='white' if abs(delta[r,c]) > .65 * scale else '#222')
                ax.set_xticks(range(3), ('.5', '1', '2'), fontsize=7)
                ax.set_yticks(range(3), ('20', '60', '100'), fontsize=7)
                if j == 0: ax.set_ylabel(SHORT[t] + '\nRRF k', fontsize=8)
                if t == 0: ax.set_title(NAMES[j], fontsize=9)
                if t == 3: ax.set_xlabel('Dense / BM25 weight', fontsize=8)
            bar = fig.colorbar(image, ax=axes[:, j], location='bottom', fraction=.035, pad=.025)
            bar.set_label('Delta from fixed reference (pp)', fontsize=7); bar.ax.tick_params(labelsize=6)
    elif exp == 'E08':
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 3.25), layout='constrained')
        delta = np.array([data[i+1, :2] - data[i, :2] for i in (0, 2, 4, 6)])
        for j, ax in enumerate(axes):
            for t in range(2):
                ax.bar(np.arange(4) + (t-.5)*.32, delta[:,t,j], width=.30, label=SHORT[t], color=COLORS[t])
            ax.set_xticks(range(4), ('Dense', 'Hybrid', 'Qwen200', 'LLM50'), rotation=25, fontsize=7)
            ax.axhline(0, color='#555', lw=.8); ax.set_ylabel('Delta ' + NAMES[j] + ' (pp)'); ax.grid(axis='y', alpha=.15)
        legend_axes = axes[0]
    elif exp == 'E09':
        from recare_baselines.ranked_eval import read_ranked, read_qrels
        fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.25), layout='constrained')
        gaps = []
        for t, task in enumerate(TASKS):
            from reproduce import data_file
            gold = read_qrels(data_file(f'qrels-{task}/test.jsonl', data_dir))
            scores = []
            for i, row_id in enumerate(ids):
                e = index[row_id, task]
                run = read_ranked(root / e['rankings'][0]['path'])
                oracle = []
                for q, judgments in gold.items():
                    if any(g not in (0, 1) for g in judgments.values()): raise ValueError('Binary oracle only')
                    positive = {d for d, g in judgments.items() if g > 0}
                    hit = len(set(run[q][:100 if i == 2 else 200]) & positive)
                    dcg = lambda n: sum(1/math.log2(k+2) for k in range(min(10,n)))
                    oracle.append(dcg(hit)/dcg(len(positive)) if positive else 0.)
                scores.append(100*np.mean(oracle)-data[i,t,0])
            gaps.append(scores)
        for j, ax in enumerate(axes.flat):
            for t in range(4):
                ax.plot(range(3), data[:,t,j] if j < 3 else gaps[t], 'o-', ms=3.5, lw=1.2,
                        color=COLORS[t], label=SHORT[t])
            ax.set_xticks(range(3), ('RRF', 'Qwen', 'Final LLM'), fontsize=7)
            ax.set_ylabel(NAMES[j]+' x100' if j<3 else 'nDCG@10 oracle gap (pp)'); ax.grid(alpha=.16)
        legend_axes = axes.flat[0]
    elif exp == 'E10':
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 3.25), layout='constrained')
        for j, (metric, ax) in enumerate(zip(METRICS, axes)):
            for t, task in enumerate(TASKS):
                for i, row_id in enumerate(ids):
                    y = [100*r['metrics'][metric] for r in per_query[row_id+'/'+task]]
                    ax.errorbar(i+(t-1.5)*.12, np.mean(y), yerr=np.std(y, ddof=1) if len(y)>1 else None,
                                color=COLORS[t], fmt='o' if i<4 else '^', capsize=2.5, ms=3.8,
                                label=SHORT[t] if i==0 else None)
            ax.set_xticks(range(5), ('Orig','Rev','Shuf','Norm','Drop*'), rotation=30, fontsize=7)
            ax.set_ylabel(NAMES[j]+' x100'); ax.grid(axis='y',alpha=.17)
        legend_axes = axes[0]
    else:
        raise ValueError(exp)
    if legend_axes is not None:
        handles, labels = legend_axes.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=len(labels), fontsize=8,
                   loc='upper center', bbox_to_anchor=(.53,1.08))
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(out / f'{exp}.{suffix}', dpi=270, bbox_inches='tight')
    plt.close(fig)
