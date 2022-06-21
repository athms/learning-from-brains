#!/usr/bin/env python3 

import os
from typing import Dict
import argparse
import numpy as np
from numpy import random
import torch
import sys
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../../')
from nit.batcher import make_batcher
from nit.tools import grab_tarfile_paths


def downstream_chance_accuracy(config: Dict=None) -> None:
    
    if config is None:
        config = vars(get_args().parse_args())

    random.seed(config["seed"])

    for dataset in ['HCP', 'ds002105']:
        dataset_dir = os.path.join(
            config['downstream_data_dir'],
            dataset
        )
        dataset_tarfile_paths = grab_tarfile_paths(dataset_dir)
        batcher = make_batcher(
            training_style='decoding',
            decoding_target='label_across_tasks' if dataset == 'HCP' else 'task_label'
        )
        validation_dataset = batcher.dataset(
            tarfiles=dataset_tarfile_paths,
            length=config["n_eval_samples"],
            
        )
        dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1
        )
    
        labels = []
        for batch_i, batch in enumerate(dataloader):
            
            if batch_i % 1000 == 0:
                print(
                    f'Loading batch: {batch_i+1} / {config["n_eval_samples"]}' 
                )

            labels.append(batch['labels'].detach().cpu().numpy().ravel())

            if batch_i >= config['n_eval_samples']-1:
                break
        
        labels = np.concatenate(labels)
        guesses = np.random.choice(
            np.unique(labels),
            len(labels),
            replace=True
        )
        acc = np.mean(labels == guesses) * 100

        print(
            f'{dataset}: Chance Acc: {acc:.3f}'
        )
        
    return None


def get_args() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description='compute downstream chance decodinga accuracies'
    )

    parser.add_argument(
        '--downstream-data-dir',
        metavar='DIR',
        default='data/tarfiles/downstream',
        type=str,
        help='path to downstream datasets '
             '(default: data/tarfiles/downstream)'
    )
    parser.add_argument(
        '--n-eval-samples',
        metavar='N',
        default=50000,
        type=int,
        help='number of eval samples to draw for evaluation '
             '(default: 50000)'
    )
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=1234,
        type=int,
        help='random seed '
             '(default: 1234)'
    )

    return parser


if __name__ == '__main__':

    trainer = downstream_chance_accuracy()
