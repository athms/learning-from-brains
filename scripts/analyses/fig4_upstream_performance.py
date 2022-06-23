#!/usr/bin/env python3

import os
import argparse
from typing import Dict
import pandas as pd
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


def fig_upstream_performance(config: Dict=None) -> None:
    """Script's main function; creates Figure 4 of the manuscript."""

    if config is None:
        config = vars(get_args().parse_args())

    os.makedirs(config['figures_dir'], exist_ok=True)

    fig, fig_axs = plt.subplot_mosaic(
        """
        ABCD
        """,
        figsize=(10, 3),
    )

    for i, (name, print_name, loss_label, ax) in enumerate(
        zip(
                ['autoencoder', 'CSM', 'BERT', 'NetBERT'],
                ['Autoencoding', 'CSM', 'Seq-BERT', 'Net-BERT'],
                [r'$L_{rec}$', r'$L_{rec}$', r'$L_{rec} + L_{cls}$', r'$L_{rec} + L_{cls}$'],
                [fig_axs['A'], fig_axs['B'], fig_axs['C'], fig_axs['D']]                
            )
        ):
        upstream_model_dir = [
            p for p in 
            os.listdir(config['upstream_models_dir'])
            if f'train-{name}' in p
            and 'warmup' not in p
            and 'Pretrained' not in p
        ]
        assert len(upstream_model_dir) == 1, \
            f'{name} should have exactly one path in ' +\
            f'{config["upstream_models_dir"]}'
        upstream_model_dir = upstream_model_dir[0]
        upstream_model_dir = os.path.join(
            config['upstream_models_dir'],
            upstream_model_dir
        )
        train_history = pd.read_csv(
            os.path.join(
                upstream_model_dir,
                'train_history.csv'
            )
        )
        eval_history = pd.read_csv(
            os.path.join(
                upstream_model_dir,
                'eval_history.csv'
            )
        )
        ax.plot(
            eval_history['step'].values,
            eval_history['loss'].values,
            label='Eval.' if i==0 else None,
            color='k',
            linestyle='solid',
            lw=2
        )
        ax.plot(
            train_history['step'].values[1:-1], # exclude 0th and final step
            train_history['loss'].values[1:-1],
            label='Train' if i==0 else None,
            color='k',
            linestyle='dashed',
            lw=1
        )
        ax.set_title(print_name)
        ax.set_xlabel('Training steps')
        ax.set_xticks((10000, 50000, 100000, 150000, 200000, 250000, 300000))
        ax.set_xticklabels((10000, '', '', '150000', '', '', 300000))
        ax.set_ylabel(f"Loss ({loss_label})")
        if i == 0:
            ax.legend(
                ncol=1,
                fontsize=8
            )
    
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            'fig4_upstream-performance.png'
        ),
        dpi=600
    )

    return None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='figure 4 of the manuscript; upstream performance of final models'
    )

    parser.add_argument(
        '--upstream-models-dir',
        metavar='DIR',
        default='results/models/upstream',
        type=str,
        help='path to directory where models are stored '
             '(default: results/models/upstream)'
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        default='results/figures',
        type=str,
        help='directory to which figure will be saved '
             '(default: results/figures)'
    )

    return parser


if __name__ == '__main__':
    fig_upstream_performance()