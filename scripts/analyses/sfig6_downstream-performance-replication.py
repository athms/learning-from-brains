#!/usr/bin/env python3
import os
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


def fig_downstream_performance(config=None) -> None:

    if config is None:
        config = vars(get_args().parse_args())

    os.makedirs(config['figures_dir'], exist_ok=True)

    assert 'replication' in os.listdir(config['downstream_models_dir']),\
        'replication not found in {}'.format(config['downstream_models_dir'])

    fig, axs = plt.subplots(
        1, 1,
        figsize=(4, 3),
    )
    architectures = ['LogisticRegression', 'autoencoder', 'GPT', 'BERT', 'NetBERT']
    labels = ['Linear Baseline', 'Autoencoding', 'CSM', 'Seq-BERT', 'Net-BERT']
    dataset = 'ds002105'
    n_train_subjects = [1,3,6,11]
    test_accuracies = np.zeros((len(architectures), len(n_train_subjects)))
    test_accuracies_replication = np.zeros((len(architectures), len(n_train_subjects)))

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
            test_acc_replication = None
            max_eval_accuracy = 0
            max_eval_accuracy_replication = 0

            for model_dir in model_dirs:
                eval_history = pd.read_csv(
                    os.path.join(
                        config['downstream_models_dir'],
                        dataset,
                        model_dir,
                        'eval_history.csv'
                    )
                )
                test_metrics = pd.read_csv(
                    os.path.join(
                        config['downstream_models_dir'],
                        dataset,
                        model_dir,
                        'test_metrics.csv'
                    )
                )

                if float(eval_history['accuracy'].values[-1]) > max_eval_accuracy:
                    test_acc = test_metrics['test_accuracy'].values[0]
                    max_eval_accuracy = float(eval_history['accuracy'].values[-1])

                replication_path = os.path.join(
                    config['downstream_models_dir'],
                    'replication',
                    dataset,
                    model_dir
                )

                if os.path.isdir(replication_path):
                    eval_history_replication = pd.read_csv(
                        os.path.join(
                            replication_path,
                            'eval_history.csv'
                        )
                    )
                    test_metrics_replication = pd.read_csv(
                        os.path.join(
                            replication_path,
                            'test_metrics.csv'
                        )
                    )

                    if float(eval_history_replication['accuracy'].values[-1]) > max_eval_accuracy_replication:
                        test_acc_replication = test_metrics_replication['test_accuracy'].values[0]
                        max_eval_accuracy_replication = float(eval_history['accuracy'].values[-1])

            if test_acc is not None:
                test_accuracies[ai, ni] = test_acc

            if test_acc_replication is not None:
                test_accuracies_replication[ai, ni] = test_acc_replication

    cbar_kws={
        "shrink": .5,
        "label": "Test accuracy (%)"
    }
    axs['A'] = sns.heatmap(
        test_accuracies_replication * 100,
        annot=True,
        fmt='.3g',
        linewidths=.5,
        cbar_kws=cbar_kws,
        ax=axs['A'],
    )
    axs['A'].set_title(f"Replication of\ndownstream adaptation to MDTB")
    axs['A'].set_xlabel('# Training subjects')
    axs['A'].set_yticklabels(labels, rotation=0, fontweight='light')
    axs['A'].set_xticklabels(n_train_subjects)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            'sfig5_downstream-performance_replication.png'
        ),
        dpi=600
    )

    return None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='appendix figure 6; replication of downstream adaptation analysis for MDTB data'
    )

    parser.add_argument(
        '--downstream-models-dir',
        metavar='DIR',
        default='results/models/downstream',
        type=str,
        help='path to directory where models are stored '
             '(default: results/models/downstream); '
             'needs to include "replication/" sub-directory, '
             'containing the model fits of the replication'
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
    fig_downstream_performance()