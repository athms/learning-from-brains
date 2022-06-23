#!/usr/bin/env python3

import os, sys
import argparse
from typing import Dict
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}')
from fig5_downstream_performance import fig_downstream_performance


def sfig_downstream_performance_replication(config: Dict=None) -> None:
    """Script's main funtion; creates Appendix Figure 6 by wrapping
    fig_downstream_performance() from scripts/analyses/fig5_downstream-performance.py"""

    if config is None:
        config = vars(get_args().parse_args())

    fig_downstream_performance(
        config=config,
        datasets=['ds002105'],
        figname='sfig6_downstream-performance-replication.png'
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
             'needs to include "replication/" directory, '
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
    sfig_downstream_performance_replication()