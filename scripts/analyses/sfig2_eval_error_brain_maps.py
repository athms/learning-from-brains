#!/usr/bin/env python3 

import os
from typing import Dict
import argparse
import json
import numpy as np
import pandas as pd
import torch
from numpy import random
from torch import manual_seed
from nilearn.regions import signals_to_img_maps 
from nilearn.datasets import fetch_atlas_difumo
import nibabel as nb
import sys
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')
from train import make_model
sys.path.insert(0, f'{script_path}/../../')
from src.batcher import make_batcher
from src.tools import plot_brain_map


def eval_error_brain_maps(config: Dict=None) -> None:
    """Script's main function; computes brain map for 
    reconstruction error of given upstream model in upstream 
    validation data"""
    
    if config is None:
        config = vars(get_args().parse_args())

    random.seed(config["seed"])
    manual_seed(config["seed"])
    os.makedirs(
        config['error_brainmaps_dir'],
        exist_ok=True
    )
    path_model_config = os.path.join(
        config["model_dir"],
        'train_config.json'
    )
    path_tarfile_paths_split = os.path.join(
        config["model_dir"],
        'tarfile_paths_split.json'
    )
    path_pretrained_model = os.path.join(
        config["model_dir"],
        'model_final',
        "pytorch_model.bin"
    )

    assert os.path.isfile(path_model_config),\
        f'{path_model_config} does not exist'
    assert os.path.isfile(path_tarfile_paths_split),\
        f'{path_tarfile_paths_split} does not exist'
    assert os.path.isfile(path_pretrained_model),\
        f'{path_pretrained_model} does not exist'

    with open(path_tarfile_paths_split, 'r') as f:
        tarfile_paths_split = json.load(f)

    with open(path_model_config, 'r') as f:
        model_config = json.load(f)

    model_config['pretrained_model'] = path_pretrained_model
    path_error_map = os.path.join(
        config['error_brainmaps_dir'],
        f'mean_eval_error_{model_config["training_style"]}.nii.gz'
    )
    
    if not os.path.isfile(path_error_map):
        batcher = make_batcher(
            training_style=model_config["training_style"],
            sample_random_seq=model_config["sample_random_seq"],
            seq_min=model_config["seq_min"],
            seq_max=model_config["seq_max"],
            bert_seq_gap_min=model_config["bert_seq_gap_min"],
            bert_seq_gap_max=model_config["bert_seq_gap_max"],
            decoding_target=model_config["decoding_target"],
            bold_dummy_mode=model_config["bold_dummy_mode"]
        )
        validation_dataset = batcher.dataset(
            tarfiles=[
                os.path.join(
                    config["data_dir"],
                    f.split('upstream/')[-1],
                )
                for f in tarfile_paths_split['validation']
            ],
            length=config["n_eval_samples"]
        )
        eval_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1
        )
        model = make_model(model_config=model_config)
        model.eval()
        mean_error = np.zeros(1024)
        network_sample_count = np.zeros(1024)
        print(
            f'\ncomputing upstream reconstruction error brain map for {model_config["training_style"]}'
        )

        for batch_i, batch in enumerate(eval_dataloader):

            if batch_i % 1000 == 0:
                print(
                    f'\tprocessing sample {batch_i} / {config["n_eval_samples"]}'
                )

            batch = {k: v[0] for k, v in batch.items()}

            with torch.no_grad():
                
                if isinstance(
                    model,
                    (
                        torch.nn.DataParallel, 
                        torch.nn.parallel.DistributedDataParallel
                    )
                ):
                    (outputs, batch) = model.module.forward(
                        batch=batch,
                        prep_batch=True,
                        return_batch=True
                    )
                
                else:
                    (outputs, batch) = model.forward(
                        batch=batch,
                        prep_batch=True,
                        return_batch=True
                    )

            if 'modelling_mask' in batch:
                # only compute error for masked inputs:
                masking_idx = batch["modelling_mask"].detach().cpu().numpy()==1
                prediction = np.zeros(batch["modelling_mask"].shape)
                prediction[masking_idx] = outputs["outputs"].detach().cpu().numpy()[masking_idx]
                inputs = np.zeros_like(prediction)
                inputs[masking_idx] = batch["masked_inputs"].detach().cpu().numpy()
                batch_error = np.absolute(prediction - inputs).sum(axis=(0, 1))
                batch_error = np.nan_to_num(batch_error / masking_idx.sum(axis=(0, 1)))
                network_idx = masking_idx.sum(axis=(0, 1)) != 0
                mean_error[network_idx] += batch_error[network_idx]
                network_sample_count[network_idx] += 1

            else:
                # error for all input values:
                prediction = outputs["outputs"].detach().cpu().numpy()
                inputs = batch["inputs"].detach().cpu().numpy()
                masking_idx = batch["attention_mask"].detach().cpu().numpy().astype(np.bool)
                batch_error = np.absolute(prediction[masking_idx] - inputs[masking_idx])
                batch_error = batch_error.mean(axis=0)
                mean_error += batch_error
                network_sample_count += 1

            if batch_i >= config['n_eval_samples']-1:
                break

        mean_error /= network_sample_count
        difumo = fetch_atlas_difumo(
            dimension=1024,
            resolution_mm=2
        )
        mean_error_df = pd.DataFrame(
            {
                'name': [l[0] for l in difumo.labels], 
                'mean_error': mean_error
            }
        )
        mean_error_df.to_csv(
            os.path.join(
                config['error_brainmaps_dir'],
                f'mean_eval_error_{model_config["training_style"]}.csv'
            ),
            index=False
        )
        error_img_map = signals_to_img_maps(
            region_signals=mean_error,
            maps_img=difumo.maps
        )
        error_img_map.to_filename(path_error_map)

    else:
        print(
            '/!\ using existing L1-error image stored in '
            f'{path_error_map}'
        )
        error_img_map = nb.load(path_error_map)

    plot_brain_map(
        img=error_img_map,
        path=os.path.join(
            config['error_brainmaps_dir'],
            f'mean_eval_error_{model_config["training_style"]}.png'
        ),
        vmin=np.min(error_img_map.get_fdata()),
        vmax=0.0035,
    )

    return None


def get_args() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description='compute upstream reconstruction error brain map for given model; '
                    'as shown in appendix figure 2 of the manuscript.'
    )

    parser.add_argument(
        '--model-dir',
        metavar='DIR',
        type=str,
        help='path to directory where model is stored '
             'for which reconstruction error brain map '
             'is to be computed.'
    )
    parser.add_argument(
        '--error-brainmaps-dir',
        metavar='DIR',
        default='results/brain_maps/l1_error',
        type=str,
        help='directory to which error brain map will be stored '
             '(default: results/brain_maps/l1_error)'
    )
    parser.add_argument(
        '--n-eval-samples',
        metavar='N',
        default=50000,
        type=int,
        help='number of random samples to draw for evaluation '
             '(default: 50000)'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        type=str,
        default='data/upstream',
        help='path to upstream data directory '
             '(default: data/upstream)'
    )
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=1234,
        type=int,
        help='random seed (default: 1234)'
    )

    return parser
2

if __name__ == '__main__':

    eval_error_brain_maps()
