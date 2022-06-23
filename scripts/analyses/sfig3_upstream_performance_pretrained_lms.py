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


def sfig_upstream_performance_pretrained_lms(config: Dict=None) -> None:
    """Script's main function; creates Appendix Figure 3 of the manuscript."""

    if config is None:
        config = vars(get_args().parse_args())

    os.makedirs(config['figures_dir'], exist_ok=True)
    fig, fig_axs = plt.subplot_mosaic(
        """
        AB
        """,
        figsize=(6, 3),
    )

    for name, print_name, loss_label, ax in zip(
        ['PretrainedGPT2', 'PretrainedBERT'],
        ['GPT2', 'BERT'],
        ['L1', 'L1 + XE'],
        fig_axs.values()        
    ):
        model_dirs = [
            p for p in 
            os.listdir(config['upstream_models_dir'])
            if name in p
        ]
        assert len(model_dirs) == 2, \
            f'{name} should have exactly two paths in {config["upstream_models_dir"]}'
        warmup_dir = [m for m in model_dirs if 'warmup' in m][0]
        train_dir = [m for m in model_dirs if 'warmup' not in m][0]
        learning_curve_start_step = 0

        for mi, model_dir in enumerate([warmup_dir, train_dir]):
            model_dir = os.path.join(
                config['upstream_models_dir'],
                model_dir
            )
            train_history = pd.read_csv(
                os.path.join(
                    model_dir,
                    'train_history.csv'
                )
            )
            eval_history = pd.read_csv(
                os.path.join(
                    model_dir,
                    'eval_history.csv'
                )
            )
            ax.plot(
                eval_history['step'].values + learning_curve_start_step,
                eval_history['loss'].values,
                label='Eval.' if mi == 1 else 'Warmup Eval.',
                color=['grey', 'k'][mi],
                linestyle='solid',
                lw=2
            )
            ax.plot(
                train_history['step'].values[1:-1] + learning_curve_start_step, # exclude 0th and final step
                train_history['loss'].values[1:-1],
                label='Train' if mi == 1 else 'Warmup Train',
                color=['grey', 'k'][mi],
                linestyle='dashed',
                lw=1
            )
            learning_curve_start_step = train_history['step'].values[-2]

        ax.set_title(print_name)
        ax.set_xlabel('Training steps')
        ax.set_xticks((0, 25000, 50000, 75000, 100000, 125000, 150000, 175000))
        ax.set_xticklabels((0, '', '', 75000, '', '', 150000, ''))
        ax.set_ylabel(f"Loss ({loss_label})")
        ax.legend(
            ncol=2,
            fontsize=8
        )
    
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            'sfig3_upstream-performance_pretrained-lms.png'
        ),
        dpi=600
    )

    return None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=' appendix figure 3 of the manuscript; upstream training performance of pre-trained language models'
    )

    parser.add_argument(
        '--upstream-models-dir',
        metavar='DIR',
        default='results/models/upstream',
        type=str,
        help='path to directory where upstream models are stored '
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
    sfig_upstream_performance_pretrained_lms()