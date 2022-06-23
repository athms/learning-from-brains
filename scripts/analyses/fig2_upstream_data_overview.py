#!/usr/bin/env python3 

import os
import argparse
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(
    context='paper',
    style="ticks",
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False
    }
)


def fig_upstream_data_overview(config: Dict=None) -> None:
    """Script's main function; creates Figure 2 of the manuscript."""

    if config is None:
        config = vars(get_args().parse_args())

    os.makedirs(
        config["figures_dir"],
        exist_ok=True
    )
    datasets = np.unique(
        [
            p for p in os.listdir(config["data_dir"])
            if os.path.isdir(os.path.join(config["data_dir"], p))
            and p.startswith('ds')
        ]
    )
    n_subjects = []
    n_runs = []
    avg_n_runs = []

    for ds in datasets:
        ds_files = [
            f for f in os.listdir(
                os.path.join(
                    config["data_dir"],
                    ds
                )
            )
            if f.endswith('.tar')
        ]
        n_runs.append(len(ds_files))
        subjects = np.unique(
            [
                f.split('sub-')[-1].split('_')[0]
                for f in ds_files
            ]
        )
        n_subjects.append(len(subjects))
        ds_avg_n_runs = []

        for subject in subjects:
            subject_files = [
                f for f in ds_files
                if f'sub-{subject}' in f
            ]
            ds_avg_n_runs.append(len(subject_files))

        avg_n_runs.append(np.mean(ds_avg_n_runs))
        print(
            f'{ds}: {len(subjects)} subjects, {len(ds_files)} runs'
        )

    n_subjects = np.asarray(n_subjects)
    avg_n_runs = np.asarray(avg_n_runs) 
    total_n_runs = np.sum([np.sum(n) for n in n_runs])
    total_n_subjects = np.sum(n_subjects)
    print(
        f'Found {total_n_subjects} subjects and {total_n_runs} runs in total'
    )

    fig, ax = plt.subplots(
        1, 1,
        figsize=(3,3),
        dpi=600
    )
    ax.scatter(
        n_subjects,
        avg_n_runs,
        s=12,
        color='k',
        alpha=0.5,
        linewidth=0
    )
    ax.set_title(f'{int(n_subjects.size)} upstream datasets')
    ax.set_ylabel('# Runs per individual')
    ax.set_xlabel('# Individuals')
    # inset axis
    x1, x2, y1, y2 = 0, 50, 0, 20
    ins_idx = np.logical_and(
        np.logical_and(
            n_subjects >= x1,
            n_subjects <= x2
        ),
        np.logical_and(
            avg_n_runs >= y1,
            avg_n_runs <= y2
        )
    )
    axins = ax.inset_axes(
        [0.5, 0.5, 0.47, 0.47]
    )
    axins.scatter(
        n_subjects[ins_idx],
        avg_n_runs[ins_idx],
        s=12,
        color='k',
        alpha=0.5,
        linewidth=0
    )
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks(
        [x1, int(x2/2.), x2]
    )
    axins.set_xticklabels(
        [x1, int(x2/2.), x2]
    )
    axins.set_yticks(
        [y1, int(y2/2.), y2]
    )
    axins.set_yticklabels(
        [y1, int(y2/2.), y2]
    )
    ax.indicate_inset_zoom(
        axins,
        edgecolor="black"
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config["figures_dir"],
            'fig2_upstream-data-overview.png'
        ),
        dpi=600
    )


def get_args() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description='figure 2 of the manuscript; overview of upstream datasets'
    )
    
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        type=str,
        default='data/upstream',
        help='path to upstream data directory '
             '(default: data/upstream)'
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        type=str,
        default='results/figures',
        help='directory to which figure will be saved '
             '(default: results/figures)'
    )

    return parser


if __name__ == '__main__':
    fig_upstream_data_overview()