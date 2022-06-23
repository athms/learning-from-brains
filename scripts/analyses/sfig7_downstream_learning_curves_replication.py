#!/usr/bin/env python3

import os, sys
import argparse
from typing import Dict
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}')
from sfig4_5_downstream_learning_curves import sfig_downstream_learning_curves


def sfig_downstream_learning_curves_replication(config: Dict=None) -> None:
    """Script's main funtion; creates Appendix Figure 7 by wrapping
    fig_downstream_performance() from scripts/analyses/fig5_downstream-performance.py"""

    if config is None:
        config = vars(get_args().parse_args())

    sfig_downstream_learning_curves(
        config=config,
        datasets=['ds002105'],
        base_figname='sfig6_downstream-learning-curves-replication'
    )

    return None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='appendix figure 7 of the manuscript; downstream learning curves for replication analysis'
    )

    parser.add_argument(
        '--downstream-models-dir',
        metavar='DIR',
        default='results/models/downstream/replication',
        type=str,
        help='path to directory where models are stored '
             '(default: results/models/downstream/replication)'
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
    sfig_downstream_learning_curves_replication()