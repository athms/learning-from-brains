#!/usr/bin/env python3
import os
import argparse
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


def sfig_downstrea_learning(config=None) -> None:

    if config is None:
        config = vars(get_argparse().parse_args())

    os.makedirs(config['figures_dir'], exist_ok=True)
    architectures = ['autoencoder', 'GPT', 'BERT', 'NetBERT', 'LogisticRegression']
    dataset = 'ds002105'
    dataset_models_dir = os.path.join(
            config['downstream_models_dir'],
            dataset
        )
    fig, axs = plt.subplots(
        5, 4,
        figsize=(9, 10),
        sharey=True
    )
    
    for ai, (architecture, name) in enumerate(
        zip(
                architectures,
                ['Autoencoding', 'CSM', 'Seq-BERT', 'Net-BERT', 'Linear Baseline']
            )
        ):
        
        training_runs = [
            p for p in 
            os.listdir(dataset_models_dir)
            if p.startswith(architecture)
        ]
        assert len(training_runs) > 0, \
            f'no training runs found for {architecture} in {dataset_models_dir}'
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

        if architecture == 'LogisticRegression':
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

            c = None
            if architecture == 'LogisticRegression':
                c = training_run.split('C-')[-1].split('_')[0]

            eval_history =  pd.read_csv(
                os.path.join(
                    dataset_models_dir,
                    training_run,
                    'eval_history.csv'
                )
            )
            color = sns.color_palette('husl')[np.where(lrs == float(lr))[0][0]]
            
            if architecture == 'LogisticRegression':
                lr_c_pairs = list(itertools.product(lrs, cs))
                _i = np.where(
                    [
                        (float(lr), float(c)) == (lr_c_pairs[i][0], lr_c_pairs[i][1])
                        for i in range(len(lr_c_pairs))
                    ]
                )[0][0]
                color = sns.color_palette('husl')[_i]

            ntrain_i = np.where(ntrains == float(ntrain))[0][0]
            axs[ai,ntrain_i].plot(
                eval_history['step'].values,
                eval_history['accuracy'].values * 100,
                label=f'LR: {lr}' if c is None else f'LR: {lr}, C: {c}',
                color=color,
                alpha=0.8,
                lw=1.5
            )
            axs[ai, ntrain_i].set_xlabel('Training steps')
        
        axs[ai, 0].set_ylabel(f"{name}\n\nEval. accuracy (%)")
        
        if architecture != 'LogisticRegression':
            patches = [
                mpatches.Patch(
                    color=sns.color_palette('husl')[i],
                    label=f'LR: {lrs[i]}'
                )
                for i in range(len(lrs))
            ]
        else:
            patches = [
                mpatches.Patch(
                    color=sns.color_palette('husl')[i],
                    label=f'LR: {lr}, '+r'$\lambda: '+f'{c}$'
                )
                for i, (lr, c) in enumerate(lr_c_pairs)
            ]
        
        if architecture == 'LogisticRegression':
            axs[ai, -1].legend(
                handles=patches,
                fontsize=6,
                loc='lower right',
                frameon=False
            )
        else:
            axs[ai, 0].legend(
                handles=patches,
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

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            f'sfig6_downstream-learning-curves_ds002105_replication.png'
        ),
        dpi=600
    )

    return None


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='supplementary figure downstream performance')

    parser.add_argument(
        '--downstream-models-dir',
        metavar='DIR',
        default='results/models/downstream/',
        type=str,
        help=''
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        default='results/figures/',
        type=str,
        help=''
    )

    return parser


if __name__ == '__main__':
    sfig_downstrea_learning()