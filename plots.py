"""Repeat-level summaries and colorblind-safe 300 dpi figures."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'WL': '#0072B2', 'P-WL': '#E69F00', 'P-WL-C': '#009E73', 'Extension': '#CC79A7'}


def table(stats, path):
    lines = ['| Dataset | Method | Accuracy (%) | Macro-F1 (%) | Repeats |',
             '|---|---|---:|---:|---:|']
    for row in stats[stats.setting == 'selected'].itertuples():
        def fmt(mean, sd):
            return f'{100*mean:.2f} ± {100*sd:.2f}' if np.isfinite(sd) else f'{100*mean:.2f} (SD unavailable)'
        lines.append(f'| {row.dataset} | {row.method} | {fmt(row.accuracy_mean, row.accuracy_std)} | '
                     f'{fmt(row.macro_f1_mean, row.macro_f1_std)} | {row.repeats} |')
    path.write_text('\n'.join(lines) + '\n')


def figures(stats, runtime, out):
    out = Path(out)
    for dataset in stats.dataset.unique():
        data = stats[stats.dataset == dataset]
        repeats = int(data.repeats.max())
        def error(sd):
            return None if sd.isna().all() else 100*sd.fillna(0)
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ['WL', 'P-WL', 'P-WL-C']:
            points = data[(data.method == method) & data.setting.str.startswith('h=')].copy()
            if points.empty:
                continue
            points['x'] = points.setting.str[2:].astype(int)
            points = points.sort_values('x')
            ax.errorbar(points.x, 100*points.accuracy_mean, yerr=error(points.accuracy_std),
                        color=COLORS[method], label=method, marker='o', capsize=3)
        ax.set(xlabel='WL depth h (iterations 0 through h)', ylabel='Accuracy (%)', title=f'{dataset}: accuracy by depth ({repeats} repeats)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{dataset}_depth.png', dpi=300)
        plt.close(fig)
        selected = data[data.setting == 'selected'].set_index('method')
        methods = [m for m in COLORS if m in selected.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        points = selected.loc[methods]
        ax.bar(methods, 100*points.accuracy_mean, yerr=error(points.accuracy_std),
               color=[COLORS[m] for m in methods], capsize=4)
        ax.set(ylabel='Accuracy (%)', title=f'{dataset}: nested method comparison ({repeats} repeats)', ylim=(0, 100))
        fig.tight_layout()
        fig.savefig(out / f'{dataset}_ablation.png', dpi=300)
        plt.close(fig)
        points = data[(data.method == 'Extension') & data.setting.str.startswith('l=')].copy()
        if not points.empty:
            points['x'] = points.setting.str[2:].astype(float)
            points = points.sort_values('x')
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, metric, label in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy (%)', 'Macro-F1 (%)']):
                ax.errorbar(points.x, 100*points[f'{metric}_mean'], yerr=error(points[f'{metric}_std']),
                            color=COLORS['Extension'], marker='o', capsize=3)
                ax.set(xlabel='Cycle weight lambda (our extension)', ylabel=label)
            fig.suptitle(f'{dataset}: cycle-weight comparison ({repeats} repeats)')
            fig.tight_layout()
            fig.savefig(out / f'{dataset}_lambda.png', dpi=300)
            plt.close(fig)
        if runtime is not None:
            points = runtime[runtime.dataset == dataset]
            fig, ax = plt.subplots(figsize=(7, 4))
            for method in points.method.unique():
                group = points[points.method == method]
                # Each p is a deterministic timed construction, not an uncertainty replicate.
                for p, series in group.groupby('p'):
                    # P-WL and P-WL-C share the measured persistence pass.
                    if method == 'P-WL-C':
                        continue
                    ax.errorbar(series.h, series.seconds,
                                yerr=series.seconds_std if 'seconds_std' in series else None,
                                marker='o', color=COLORS[method], capsize=3,
                                linestyle='-' if p in (0,1) else '--',
                                label=('P-WL / P-WL-C (shared)' if method=='P-WL' else method) + (f' (p={p})' if p else ''))
            timing_label = ' (3 fresh trials)' if 'seconds_std' in points else ''
            ax.set(xlabel='WL depth h', ylabel='Cumulative feature construction (seconds)', title=f'{dataset}: feature runtime by depth{timing_label}')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f'{dataset}_runtime.png', dpi=300)
            plt.close(fig)
