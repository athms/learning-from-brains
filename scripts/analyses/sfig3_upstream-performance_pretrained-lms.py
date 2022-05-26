#!/usr/bin/env python3
import os
import argparse
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


def fig_upstream_performance_pretrained_lms(config=None) -> None:

    if config is None:
        config = vars(get_argparse().parse_args())

    os.makedirs(config['figures_dir'], exist_ok=True)

    fig, fig_axs = plt.subplot_mosaic(
        """
        AB
        """,
        figsize=(6, 3),
    )

    model_names = ['PretrainedGPT2', 'PretrainedBERT']
    for i, (model_name, ax, loss_label, name) in enumerate(
        zip(
                model_names,
                fig_axs.values(),
                ['L1', 'L1 + XE'],
                ['GPT2', 'BERT']
            )
        ):
        model_dirs = [
            p for p in 
            os.listdir(config['upstream_models_dir'])
            if model_name in p
        ]
        assert len(model_dirs) == 2, \
            f'{model_name} should have exactly two paths in {config["upstream_models_dir"]}'
        warmup_dir = [m for m in model_dirs if 'warmup' in m][0]
        train_dir = [m for m in model_dirs if 'warmup' not in m][0]
        
        start_step = 0
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
                eval_history['step'].values + start_step,
                eval_history['loss'].values,
                label=f'Eval.' if mi == 1 else f'Warmup Eval.',
                color=['grey', 'k'][mi],
                linestyle='solid',
                lw=2
            )
            ax.plot(
                train_history['step'].values[1:-1] + start_step, # exclude 0th and final step
                train_history['loss'].values[1:-1],
                label=f'Train' if mi == 1 else f'Warmup Train',
                color=['grey', 'k'][mi],
                linestyle='dashed',
                lw=1
            )
            start_step = train_history['step'].values[-2]

        ax.set_title(f"{name}")
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


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='figure hyperopt')

    parser.add_argument(
        '--upstream-models-dir',
        metavar='DIR',
        default='results/models/hyperopt/',
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
    fig_upstream_performance_pretrained_lms()