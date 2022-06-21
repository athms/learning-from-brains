#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import itertools
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


def sfig_downstream_learning(config=None) -> None:

    if config is None:
        config = vars(get_args().parse_args())

    os.makedirs(config['figures_dir'], exist_ok=True)

    fig, fig_axs = plt.subplot_mosaic(
        """
        ACEG
        BDFH
        """,
        figsize=(10, 5),
    )

    linestyles = ['solid', 'dashed', 'dotted', 'dashdot', ]

    architectures = ['autoencoder', 'GPT', 'BERT', 'NetBERT']
    for ai, (architecture, axs, loss_label, name) in enumerate(
        zip(
                architectures,
                [
                    [fig_axs['A'], fig_axs['B']],
                    [fig_axs['C'], fig_axs['D']],
                    [fig_axs['E'], fig_axs['F']],
                    [fig_axs['G'], fig_axs['H']],
                ],
                [r'$L_{rec}$', r'$L_{rec}$', r'$L_{rec} + L_{cls}$', r'$L_{rec} + L_{cls}$'],
                ['Autoencoding', 'CSM', 'Seq-BERT', 'Net-BERT']
            )
        ):

        for di, dataset in enumerate(['HCP', 'ds002105']):
            dataset_models_dir = os.path.join(
                config['downstream_models_dir'],
                dataset
            )
            training_runs = [
                p for p in 
                os.listdir(dataset_models_dir)
                if p.startswith(architecture)
            ]
            assert len(training_runs) > 0, \
                f'no training runs found for {architecture} in {dataset_models_dir}'
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
                            int(p.split('C-')[-1].split('_')[0])
                            for p in training_runs
                        ]
                    )
                )
                
            for training_run in training_runs:
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
                color = sns.color_palette()[np.where(lrs == float(lr))[0][0]]
                
                if architecture == 'LogisticRegression':
                    lr_c_pairs = c = list(itertools.product(lrs, cs))
                    color = sns.color_palette()[np.where(lr_c_pairs == ( float(lr), int(c) ))[0][0]]

                axs[di].plot(
                    eval_history['step'].values,
                    eval_history['accuracy'].values,
                    label=f'LR: {lr}' if c is None else f'LR: {lr}, C: {c}',
                    color=color,
                    alpha=0.5
                )
        
            axs[di].set_title(f"{name}")
            axs[di].set_ylabel(f"Eval. loss ({loss_label})")
            axs[di].set_xlabel('Training steps')
            if ai == 0:
                ds_name = 'MDTB' if dataset == 'ds002105' else dataset
                axs[di].set_ylabel(f"Dowsntream dataset {di+1}: {ds_name} \n\n Eval. loss ({loss_label})")
            
            if architecture != 'LogisticRegression':
                patches = [
                    mpatches.Patch(
                        color=sns.color_palette()[i],
                        label=f'LR: {lrs[i]}'
                    )
                    for i in range(len(lrs))
                ]
            else:
                patches = [
                    mpatches.Patch(
                        color=sns.color_palette()[i],
                        label=f'LR: {lr}, C: {c}'
                    )
                    for i, (lr, c) in enumerate(lr_c_pairs)
                ]
            
            plt.legend(
                handles=patches,
                fontsize=6.5
            )
        
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            'sfig4_downstream-learning-curves.png'
        ),
        dpi=600
    )

    return None


def get_args() -> argparse.ArgumentParser:
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
    sfig_downstream_learning()