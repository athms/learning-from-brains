#!/usr/bin/env python3

import os
import argparse
from typing import Dict
import pandas as pd
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


def fig_hyperopt(config: Dict=None) -> None:
    """Script's main function; creates Figure 3 of the manuscript."""

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

    for hi, (name, print_name, loss_label, axs) in enumerate(
        zip(
                ['autoencoder', 'CSM', 'BERT', 'NetBERT'],
                ['Autoencoding', 'CSM', 'Seq-BERT', 'Net-BERT'],
                [r'$L_{rec}$', r'$L_{rec}$', r'$L_{rec} + L_{cls}$', r'$L_{rec} + L_{cls}$'],
                [
                    [fig_axs['A'], fig_axs['B']],
                    [fig_axs['C'], fig_axs['D']],
                    [fig_axs['E'], fig_axs['F']],
                    [fig_axs['G'], fig_axs['H']],
                ]
            )
        ):
        hyperopt_path = [
            p for p in 
            os.listdir(config['hyperopt_dir'])
            if f'-{name}' in p
        ]
        assert len(hyperopt_path) == 1, \
            f'{name} should have exactly one path in {config["hyperopt_dir"]}'
        hyperopt_path = hyperopt_path[0]
        hyperopt_path = os.path.join(
            config['hyperopt_dir'],
            hyperopt_path
        )
        model_paths = [
            os.path.join(hyperopt_path, p)
            for p in os.listdir(hyperopt_path)
            if 'lrs-' in p and 'embd-' in p
        ]
        n_layers = np.unique(
            [
                int(p.split('/')[-1].split('lrs-')[1].split('_')[0])
                for p in model_paths
            ]
        )
        n_layers = np.sort(n_layers)
        embedding_dims = np.unique(
            [
                int(p.split('/')[-1].split('embd-')[1].split('_')[0])
                for p in model_paths
            ]
        )
        embedding_dims = np.sort(embedding_dims)
        final_eval_loss = np.zeros((len(n_layers), len(embedding_dims)))
        linestyles = ['solid', 'dashed', 'dotted']

        for li, l in enumerate(n_layers):

            for ei, e in enumerate(embedding_dims):
                model_path = [
                    p for p in model_paths
                    if f'lrs-{l}' in p and f'embd-{e}' in p
                ]
                assert len(model_path) == 1,\
                    f'Found {len(model_path)} instead of 1 '+\
                    f'model for with lrs-{l} and embd-{e}'
                model_path = model_path[0]
                eval_history = pd.read_csv(
                    os.path.join(
                        model_path,
                        'eval_history.csv'
                    )
                )
                assert eval_history['step'].values[-1]==200000,\
                    f'Last step in eval_history.csv is {eval_history["step"].values[-1]}, '+\
                    'but should be 200000'
                final_eval_loss[li, ei] = eval_history['loss'].values[-1]

                # we are plotting the evaluation loss of the 
                # largest Seq-BERT model variant separately;
                # see sfig_training-curve-larges-sequence-BERT.py
                if l==12 and e ==768 and name=='BERT':
                    continue

                axs[0].plot(
                    eval_history['step'].values,
                    eval_history['loss'].values,
                    label=f'{l}, {e}',
                    color=sns.color_palette("Set2")[ei],
                    linestyle=linestyles[li]
                )

        cbar_kws={
            "shrink": .5,
            "label": f"{loss_label}"
        }
        axs[1] = sns.heatmap(
            final_eval_loss,
            square=True,
            annot=True,
            fmt='.2g',
            linewidths=.5,
            cmap=sns.cm.rocket_r,
            cbar_kws=cbar_kws,
            annot_kws={'size': 8},
            ax=axs[1],
        )
        axs[0].set_title(print_name)
        axs[0].set_xlabel('Training steps')
        axs[0].set_xticks((0, 50000, 100000, 150000, 200000))
        axs[0].set_xticklabels((0, '', 100000, '', 200000))
        axs[0].set_ylabel(f"Eval. loss ({loss_label})")
        axs[1].set_title('Final eval. loss')
        axs[1].set_yticklabels(
            n_layers,
            rotation=0,
            fontweight='light'
        )
        axs[0].legend(
            loc='upper right',
            ncol=2,
            fontsize=6.5,
            bbox_to_anchor=(1.1, 1.06)
        )
        
        if hi == 0:
            axs[1].set_ylabel('# Layers')
        
        if name != 'autoencoder':
            axs[1].set_xlabel('Embedding dim.\n(# Attn. heads)')
            axs[1].set_xticklabels(
                [f'{e}\n({e//64})' for e in embedding_dims],
                rotation=0,
                fontweight='light'
            )
            
        else:
            axs[1].set_xlabel('Embedding dim.')
            axs[1].set_xticklabels(
                [f'{e}' for e in embedding_dims],
                rotation=0,
                fontweight='light'
            )
    
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            'fig3_hyperopt.png'
        ),
        dpi=600
    )

    return None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='figure 3 of the manuscript; hyper-optimization model performances'
    )

    parser.add_argument(
        '--hyperopt-dir',
        metavar='DIR',
        default='results/models/hyperopt',
        type=str,
        help='path to directory where hyper-optimization results are stored '
             '(default: results/models/hyperopt)'
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
    fig_hyperopt()