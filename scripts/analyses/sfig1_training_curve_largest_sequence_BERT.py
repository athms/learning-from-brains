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


def sfig_training_curve_seqBERT_large(config: Dict=None) -> None:
    """Script's main function; creates Appendix Figure 1 of the manuscript."""

    if config is None:
        config = vars(get_args().parse_args())

    os.makedirs(config['figures_dir'], exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(3,3),
        dpi=600
    )
    bert_hyperopt_dir = os.path.join(
        config['hyperopt_dir'],
        "BERT-BERT",
    )
    model_dir = [
        p for p in 
        os.listdir(bert_hyperopt_dir)
        if 'lrs-12' in p and 'embd-768' in p
    ][0]
    eval_history = pd.read_csv(
        os.path.join(
            bert_hyperopt_dir,
            model_dir,
            "eval_history.csv"
        )
    )
    ax.plot(
        eval_history['step'].values,
        eval_history['loss'].values,
        color='k',
        linewidth=1
    )
    ax.set_title("Sequence-BERT\n(12 layers, 768 dim.)")
    ax.set_xlabel('Training steps')
    ax.set_xticks((0, 50000, 100000, 150000, 200000))
    ax.set_xticklabels((0, '', 100000, '', 200000))
    loss_label = r'$L_{rec} + L_{cls}$'
    ax.set_ylabel(f"Eval. loss ({loss_label})")
    ax.set_ylim(0.8, 1.6)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            'sfig1_training-curve-largest-sequence-BERT.png'
        ),
        dpi=600
    )

    return None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='appendix figure 1; training performance of largest Seq-BERT model variant'
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
        default='results/figures',
        type=str,
        help='directory to which figure will be saved '
             '(default: results/figures)'
    )

    return parser


if __name__ == '__main__':
    sfig_training_curve_seqBERT_large()