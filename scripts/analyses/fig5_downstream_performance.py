#!/usr/bin/env python3

import os
from typing import Dict, Tuple
import argparse
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


def fig_downstream_performance(
    config: Dict=None,
    datasets: Tuple[str, ...]=('HCP', 'ds002105'),
    figname: str='fig5_downstream-performance.png'
    ) -> None:
    """Script's main function; creates Figure 5 of the manuscript."""

    if config is None:
        config = vars(get_args().parse_args())

    for ds in datasets:
        assert ds in os.listdir(config['downstream_models_dir']),\
            f'{ds} not found in {config["downstream_models_dir"]}'
        assert ds in ['HCP', 'ds002105'], \
            f'{ds} not supported; must be one of HCP or ds002105'

    n_ds = len(datasets)
    os.makedirs(config['figures_dir'], exist_ok=True)
    moisaic = [[f'ds{i}'] for i in range(n_ds)]
    fig, axs = plt.subplot_mosaic(
        moisaic,
        figsize=(4*n_ds, 3),
    )
    architectures = ['LinearBaseline', 'autoencoder', 'GPT', 'BERT', 'NetBERT']
    training_styles = ['Linear Baseline', 'Autoencoding', 'CSM', 'Seq-BERT', 'Net-BERT']

    for di, dataset in enumerate(datasets):
        n_train_subjects = [1,3,6,12,24,48] if dataset == 'HCP' else [1,3,6,11]
        test_accuracies = np.zeros((len(architectures), len(n_train_subjects)))

        for ai, architecture in enumerate(architectures):
            
            for ni, n_train in enumerate(n_train_subjects):
                model_dirs = [
                    p for p in 
                    os.listdir(
                        os.path.join(
                            config['downstream_models_dir'],
                            dataset
                        )
                    )
                    if p.startswith(architecture)
                    and f'ntrain-{n_train}_' in p
                ]
                test_acc = None
                max_eval_accuracy = 0

                for model_dir in model_dirs:
                    eval_history = pd.read_csv(
                        os.path.join(
                            config['downstream_models_dir'],
                            dataset,
                            model_dir,
                            'eval_history.csv'
                        )
                    )

                    if float(eval_history['accuracy'].values[-1]) > max_eval_accuracy:
                        test_acc = pd.read_csv(
                            os.path.join(
                                config['downstream_models_dir'],
                                dataset,
                                model_dir,
                                'test_metrics.csv'
                            )
                        )['test_accuracy'].values[0]
                        max_eval_accuracy = float(eval_history['accuracy'].values[-1])
                
                if test_acc is not None:
                    test_accuracies[ai, ni] = test_acc
                
        cbar_kws={
            "shrink": .5,
            "label": "Test accuracy (%)"
        }
        axs[f'ds{di}'] = sns.heatmap(
            test_accuracies * 100,
            annot=True,
            fmt='.3g',
            linewidths=.5,
            cbar_kws=cbar_kws,
            ax=axs[f'ds{di}'],
        )
        title = dataset if dataset == 'HCP' else 'MDTB'
        axs[f'ds{di}'].set_title(f"Downstream dataset {di+1} ({title})")
        axs[f'ds{di}'].set_xlabel('# Training subjects')
        if dataset == 'HCP':
            axs[f'ds{di}'].set_yticklabels(training_styles, rotation=0, fontweight='light')
        else:
            axs[f'ds{di}'].set_yticklabels(['' for _ in range(len(architectures))])
        axs[f'ds{di}'].set_xticklabels(n_train_subjects)
        
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            figname
        ),
        dpi=600
    )

    return None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='figure 5 of the manuscript; downstream model adapatation performances'
    )

    parser.add_argument(
        '--downstream-models-dir',
        metavar='DIR',
        default='results/models/downstream',
        type=str,
        help='path to directory where downstream models are stored '
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
    fig_downstream_performance()