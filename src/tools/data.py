#!/usr/bin/env python3

import os
import numpy as np
from typing import Tuple


def grab_tarfile_paths(path) -> Tuple[str]:
    paths = os.listdir(path)
    tarfiles = []

    for p in paths:

        if os.path.isdir(
            os.path.join(
                path,
                p
            )
        ):
            tarfiles += [
                os.path.join(
                    path,
                    p,
                    f
                )
                for f in os.listdir(
                    os.path.join(
                        path,
                        p
                    )
                )
                if f.endswith('.tar')
            ]

        elif np.logical_and(
            os.path.isfile(
                os.path.join(
                    path,
                    p
                )
            ), 
            p.endswith('.tar')
        ):
            tarfiles.append(
                os.path.join(
                    path,
                    p
                )
            )

    return sorted(np.unique(tarfiles))


def split_tarfile_paths_train_val(
    tarfile_paths,
    frac_val_per_dataset: float=0.05,
    n_val_subjects_per_dataset: int=None,
    n_test_subjects_per_dataset: int=None,
    n_train_subjects_per_dataset: int=None,
    min_val_per_dataset: int=2,
    seed: int=1234
    ) -> Tuple[str]:
    np.random.seed(seed)
    datasets = np.unique(
        [
            f.split('/')[-1].split('ds-')[1].split('_')[0]
            for f in tarfile_paths
        ]
    )
    train_tarfiles, val_tarfiles = [], []
    test_tarfiles = [] if n_test_subjects_per_dataset is not None else None
    
    for dataset in datasets:
        dataset_tarfiles = np.unique(
            [
                f for f in tarfile_paths
                if f'ds-{dataset}' in f
            ]
        )

        if n_val_subjects_per_dataset is None and \
           n_test_subjects_per_dataset is None and \
           n_train_subjects_per_dataset is None:
            np.random.shuffle(dataset_tarfiles)
            n_val = max(
                int(len(dataset_tarfiles)*frac_val_per_dataset),
                min_val_per_dataset
            )
            train_tarfiles += list(dataset_tarfiles[:-n_val])
            val_tarfiles += list(dataset_tarfiles[-n_val:])

        else:
            subjects = np.unique(
                [
                    f.split('_sub-')[1].split('_')[0]
                    for f in dataset_tarfiles
                ]
            )
            n_test_subjects_per_dataset = 0 if n_test_subjects_per_dataset is None else n_test_subjects_per_dataset
            assert n_val_subjects_per_dataset is not None,\
                'n_train_subjects_per_dataset and n_val_subjects_per_dataset must be specified'
            assert n_val_subjects_per_dataset < len(subjects),\
                'n_val_subjects_per_dataset must be smaller than the number of subjects'
            n_train_subjects_per_dataset = len(subjects)-n_val_subjects_per_dataset if n_train_subjects_per_dataset is None else n_train_subjects_per_dataset
            assert (
                n_val_subjects_per_dataset+\
                n_test_subjects_per_dataset+\
                n_train_subjects_per_dataset
            ) <= len(subjects), \
                f'Not enough subjects in dataset {dataset} for '\
                f'{n_val_subjects_per_dataset} val, '\
                f'{n_test_subjects_per_dataset} test, '\
                f'{n_train_subjects_per_dataset} train'

            validation_subjects = np.random.choice(
                subjects,
                n_val_subjects_per_dataset,
                replace=False
            )
            if n_test_subjects_per_dataset > 0:
                test_subjects = np.random.choice(
                    [s for s in subjects if s not in validation_subjects],
                    n_test_subjects_per_dataset,
                    replace=False
                )
            else:
                test_subjects = []

            train_subjects = [
                s for s in subjects
                if s not in validation_subjects
                and s not in test_subjects
            ][:n_train_subjects_per_dataset]

            for subject in subjects:
                
                if subject in validation_subjects:
                    val_tarfiles += [
                        f for f in dataset_tarfiles
                        if f'sub-{subject}' in f
                    ]
                
                elif subject in train_subjects:
                    train_tarfiles += [
                        f for f in dataset_tarfiles
                        if f'sub-{subject}' in f
                    ]
                
                elif subject in test_subjects:
                    test_tarfiles += [
                        f for f in dataset_tarfiles
                        if f'sub-{subject}' in f
                    ]

                else:
                    continue
    
    if test_tarfiles is None:
        return {
            'train': train_tarfiles,
            'validation': val_tarfiles,
        }
    
    else:
        return {
            'train': train_tarfiles,
            'validation': val_tarfiles,
            'test': test_tarfiles
        }