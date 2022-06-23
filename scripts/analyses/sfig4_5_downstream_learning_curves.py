#!/usr/bin/env python3

import os
import argparse
from typing import Dict, Tuple 
import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_theme(
    context='paper',
    style="ticks",
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False
    }
)


def sfig_downstream_learning_curves(
    config: Dict=None,
    datasets: Tuple[str, ...]=('HCP', 'ds002105'),
    base_figname: str='sfig_downstream-learning-curves'
    ) -> None:

    if config is None:
        config = vars(get_argparse().parse_args())

    for ds in datasets:
        assert ds in os.listdir(config['downstream_models_dir']),\
            f'{ds} not found in {config["downstream_models_dir"]}'
        assert ds in ['HCP', 'ds002105'], \
            f'{ds} not supported; must be one of HCP or ds002105'

    for dataset in datasets:
        print(
            f'\nCreating figure for {dataset} dataset'
        )
        dataset_models_dir = os.path.join(
                config['downstream_models_dir'],
                dataset
            )

        if dataset == 'HCP':
            fig, axs = plt.subplots(
                5, 6,
                figsize=(10, 10),
                sharey=True
            )

        else:
            fig, axs = plt.subplots(
                5, 4,
                figsize=(9, 10),
                sharey=True
            )

        for i, (name, print_name) in enumerate(
            zip(
                    ['autoencoder', 'GPT', 'BERT', 'NetBERT', 'LinearBaseline'],
                    ['Autoencoding', 'CSM', 'Seq-BERT', 'Net-BERT', 'Linear Baseline']
                )
            ):
            training_runs = [
                p for p in 
                os.listdir(dataset_models_dir)
                if p.startswith(name)
            ]
            assert training_runs,\
                f'no training runs found for {name} in {dataset_models_dir}'
            ntrains = np.sort(
                np.unique(
                    [
                        float(p.split('ntrain-')[-1].split('_')[0])
                        for p in training_runs
                    ]
                )
            )
            lrs = np.sort(
                np.unique(
                    [
                        float(p.split('lr-')[-1].split('_')[0])
                        for p in training_runs
                    ]
                )
            )

            if name == 'LinearBaseline':
                # also include L2-regularisation strength (C)
                cs = np.sort(
                    np.unique(
                        [
                            float(p.split('C-')[-1].split('_')[0])
                            for p in training_runs
                        ]
                    )
                )
                
            for training_run in training_runs:
                ntrain = training_run.split('ntrain-')[-1].split('_')[0]
                lr = training_run.split('lr-')[-1].split('_')[0]
                c = None # make sure C is reset for every run

                if name == 'LinearBaseline':
                    c = training_run.split('C-')[-1].split('_')[0]

                eval_history =  pd.read_csv(
                    os.path.join(
                        dataset_models_dir,
                        training_run,
                        'eval_history.csv'
                    )
                )
                color = sns.color_palette('husl')[np.where(lrs == float(lr))[0][0]]
                
                if name == 'LinearBaseline':
                    lr_c_pairs = list(itertools.product(lrs, cs))
                    color_i = np.where(
                        [
                            (float(lr), float(c)) == (lr_c_pairs[i][0], lr_c_pairs[i][1])
                            for i in range(len(lr_c_pairs))
                        ]
                    )[0][0]
                    color = sns.color_palette('husl')[color_i]

                ntrain_i = np.where(ntrains == float(ntrain))[0][0]
                axs[i, ntrain_i].plot(
                    eval_history['step'].values,
                    eval_history['accuracy'].values * 100,
                    label=f'LR: {lr}' if c is None else f'LR: {lr}, C: {c}',
                    color=color,
                    alpha=0.8,
                    lw=1.5
                )
                axs[i, ntrain_i].set_xlabel('Training steps')
            
            axs[i, 0].set_ylabel(f"{print_name}\n\nEval. accuracy (%)")
            
            if name != 'LinearBaseline':
                legend_patches = [
                    mpatches.Patch(
                        color=sns.color_palette('husl')[i],
                        label=f'LR: {lrs[i]}'
                    )
                    for i in range(len(lrs))
                ]
            
            else:
                legend_patches = [
                    mpatches.Patch(
                        color=sns.color_palette('husl')[i],
                        label=f'LR: {lr}, '+r'$\lambda: '+f'{c}$'
                    )
                    for i, (lr, c) in enumerate(lr_c_pairs)
                ]
            
            if dataset == 'ds002105' and name == 'LinearBaseline':
                axs[i, -1].legend(
                    handles=legend_patches,
                    fontsize=6,
                    loc='lower right',
                    frameon=False
                )

            else:
                axs[i, 0].legend(
                    handles=legend_patches,
                    fontsize=7,
                    loc='upper left',
                    frameon=False
                )
        
        for ni, ntrain in enumerate(ntrains):
            axs[0, ni].set_title(f'# Train subs: {int(ntrain)}')

        for ax in axs.ravel():
            ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax.grid(b=True, which='major', color='w', linewidth=1.5)
            ax.grid(b=True, which='minor', color='w', linewidth=0.75)
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')    
            ax.set_ylim(0, 100)

        os.makedirs(config['figures_dir'], exist_ok=True) 
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                config['figures_dir'],
                f'{base_figname}_{dataset}.png'
            ),
            dpi=600
        )

    return None


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='appendix figure 4-5 of the manuscript; downstream adapatation learning curves'
    )

    parser.add_argument(
        '--downstream-models-dir',
        metavar='DIR',
        default='results/models/downstream/',
        type=str,
        help='path to directory where models are stored '
             '(default: results/models/downstream)'
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        default='results/figures/',
        type=str,
        help='directory to which figure will be saved '
             '(default: results/figures)'
    )

    return parser


if __name__ == '__main__':
    sfig_downstream_learning_curves()