#!/usr/bin/env python3

import os
from typing import List, Union
import tarfile
import numpy as np
import pandas as pd
import argparse
from shutil import copyfile


def hcp_to_bids(args: argparse.Namespace=None) -> None:

    if args is None:
        args = get_args()

    subjects = np.unique(os.listdir(args.source_dir))
    tasks = [
        'EMOTION',
        'GAMBLING',
        'LANGUAGE',
        'MOTOR',
        'RELATIONAL',
        'SOCIAL',
        'WM',
        'REST1',
        'REST2'
    ]
    runs = [
        'LR',
        'RL'
    ]

    for subject in subjects:
        sub_source_dir = os.path.join(
            args.source_dir,
            subject,
            'unprocessed',
            '3T'
        )
        # anat 
        source_path = os.path.join(
            sub_source_dir,
            'T1w_MPR1',
            '{}_3T_T1w_MPR1.nii.gz'.format(subject)
        )
        target_path = os.path.join(
            args.target_dir,
            'sub-{}'.format(subject),
            'anat'
        )
        os.makedirs(
            target_path,
            exist_ok=True
        )
        target_path = os.path.join(
            target_path,
            'sub-{}_T1w.nii.gz'.format(subject)
        )

        if os.path.isfile(source_path):
            
            if not os.path.isfile(target_path):
                print(f'copying {source_path} to {target_path}')
                copyfile(
                    source_path,
                    target_path
                )
            
            else:
                print(f'/!\ {target_path} exists already.')
        
        else:
            print(f'/!\ {source_path} does not exist')
        
        for task in tasks:
            
            for run_i, run in enumerate(runs, start=1):
                
                # func data
                if task in {'REST1', 'REST2'}:
                    source_path = os.path.join(
                        sub_source_dir,
                        f'rfMRI_{task}_{run}',
                        f'{subject}_3T_rfMRI_{task}_{run}.nii.gz'
                    )
                
                else:
                    source_path = os.path.join(
                        sub_source_dir,
                        f'tfMRI_{task}_{run}',
                        f'{subject}_3T_tfMRI_{task}_{run}.nii.gz'
                    )

                target_path = os.path.join(
                    args.target,
                    f'sub-{subject}',
                    'func'
                )
                os.makedirs(
                    target_path,
                    exist_ok=True
                )
                target_path = os.path.join(
                    target_path,
                    f'sub-{subject}_task-{task}_run-{run_i}_bold.nii.gz'
                )
                
                if os.path.isfile(source_path):
                    
                    if not os.path.isfile(target_path):
                        print(
                            'copying {} to {}'.format(
                                source_path,
                                target_path
                            )
                        )
                        copyfile(
                            source_path,
                            target_path
                        )
                        
                    else:
                        print(f'/!\ {target_path} exists already.')
                
                else:
                    print(f'/!\ {source_path} does not exist')

                # ev data
                if task in {'REST1', 'REST2'}:
                    continue
                
                tar_path = os.path.join(
                    args.source_dir,
                    subject,
                    'MNINonLinear',
                    'Results',
                    f'tfMRI_{task}_{run}.tar.gz'
                )
                
                if os.path.isfile(tar_path):
                    
                    with tarfile.open(name=tar_path, mode='r') as tar:
                                            
                        for f in evs_for_task(task):

                            try:
                                source_path = tar.extractfile(f'tfMRI_{task}_{run}/EVs/{f}')
                            
                            except KeyError:
                                print(
                                    f'/!\ {f} does not exist in {tar_path}'
                                )
                                continue

                            target_path = os.path.join(
                                args.target_dir,
                                f'sub-{subject}',
                                'func',
                                f'sub-{subject}_task-{task}_run-{run_i}_EV_{f}'
                            )
                        
                            if not os.path.isfile(target_path):
                                print(
                                    f'Writing {target_path}'
                                )
                                
                                try:
                                    source_dat = pd.read_csv(
                                        source_path,
                                        sep='\t',
                                        header=None
                                    )
                                    source_dat.to_csv(
                                        target_path,
                                        sep='\t',
                                        header=False,
                                        index=False
                                    )
                                
                                except pd.errors.EmptyDataError:
                                    print(
                                        f'/!\ tfMRI_{task}_{run}.tar.gz/EVs/{f} is empty'
                                    )
                                    continue

                            else:
                                print(
                                    '/!\ {} exists already'.format(target_path)
                                )
                    
                    target_path = os.path.join(
                        args.target_dir,
                        f'sub-{subject}',
                        'func',
                        f'sub-{subject}_task-{task}_run-{run_i}_EV.csv'
                    )
                    
                    if not os.path.isfile(target_path):
                        print(
                            'creating {}'.format(target_path)
                        )
                        ev_filepaths = [
                            os.path.join(
                                args.target_dir,
                                f'sub-{subject}',
                                'func',
                                f'sub-{subject}_task-{task}_run-{run_i}_EV_{f}'
                            )
                            for f in evs_for_task(task)
                        ]
                        
                        ev_df = summarize_evs(
                            ev_filepaths=ev_filepaths,
                            task=task,
                            subject=subject,
                            run=run
                        )
                        
                        if isinstance(ev_df, pd.DataFrame):
                            ev_df.to_csv(
                                target_path,
                                index=False
                            )

                    else:
                        print(f'/!\ {target_path} exists already')


def identify_event_type_from_filepath(
    filepath: str,
    task: str
    ) -> str:
    
    if (
            task == 'GAMBLING' and
            '_event' in filepath
            or task == 'SOCIAL' and
            '_resp' in filepath
        ):
        return None

    event_type = filepath.split('.')[0].split('EV_')[-1]

    if task == 'SOCIAL' and 'resp' in event_type:
        return event_type.split('_')[-2]

    elif task == 'WM':
        
        event_type = event_type.split('_')[-1]
        
        if event_type in {
                'body',
                'faces',
                'places',
                'tools'
            }:
            return event_type
        
        else:
            return None

    else:
        return event_type.split('_')[0]


def summarize_evs(
    ev_filepaths: List[str],
    task: str,
    subject: Union[int, str],
    run: Union[int, str]
    ) -> pd.DataFrame:

    if task not in {
        'EMOTION',
        'SOCIAL',
        'GAMBLING',
        'LANGUAGE',
        'RELATIONAL',
        'MOTOR',
        'WM'
        }:
        raise NameError('Invalid task type.')

    df_list = []

    for filepath in ev_filepaths:

        event_type = identify_event_type_from_filepath(
            filepath=filepath,
            task=task
        )
        if event_type is None:
            continue

        if 'cue' not in event_type:

            try:
                ev_mat = pd.read_csv(filepath, sep='\t', header=None).values

            except pd.errors.EmptyDataError:
                print(
                    f'/!\ {filepath} because is empty.'
                )
                continue

            except FileNotFoundError:
                print(
                    f'/!\ {filepath} does not exist'
                )
                continue

            df_tmp = pd.DataFrame(
                {
                    'subject': subject,
                    'task': task,
                    'run': run,
                    'event_type': event_type,
                    'onset': ev_mat[:, 0],
                    'duration': ev_mat[:, 1],
                    'end': ev_mat[:, 0] + ev_mat[:, 1]
                }
            )
            df_list.append(df_tmp)
    
    return pd.concat(df_list) if df_list else None


def evs_for_task(
    task: str
    ) -> List[str]:

    if task == 'EMOTION':
        file_types = [
            'fear.txt',
            'neut.txt'
        ]

    elif task == 'GAMBLING':
        file_types = [
            'win.txt',
            'loss.txt',
            'win_event.txt',
            'loss_event.txt',
            'neut_event.txt'
        ]

    elif task == 'LANGUAGE':
        file_types = [
            'story.txt',
            'math.txt'
        ]

    elif task == 'MOTOR':
        file_types = [
            'cue.txt',
            'lf.txt',
            'rf.txt',
            'lh.txt',
            'rh.txt',
            't.txt'
        ]

    elif task == 'RELATIONAL':
        file_types = [
            'relation.txt',
            'match.txt'
        ]

    elif task == 'SOCIAL':
        file_types = [
            'mental.txt',
            'rnd.txt',
            'mental_resp.txt',
            'other_resp.txt'
        ]

    elif task == 'WM':
        file_types = [
            '0bk_body.txt',
            '0bk_faces.txt',
            '0bk_places.txt',
            '0bk_tools.txt',
            '2bk_body.txt',
            '2bk_faces.txt',
            '2bk_places.txt',
            '2bk_tools.txt',
            '0bk_cor.txt',
            '0bk_err.txt',
            '0bk_nlr.txt',
            '2bk_cor.txt',
            '2bk_err.txt',
            '2bk_nlr.txt',
            'all_bk_cor.txt',
            'all_bk_err.txt'
        ]

    else:
        file_types = None
        raise NameError('Invalid task type.')
    
    return file_types


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='hcp to BIDS')
    parser.add_argument(
        '--source-dir',
        metavar='DIR',
        type=str,
        help='path to HCP source directory'
    )
    parser.add_argument(
        '--target-dir',
        metavar='DIR',
        type=str,
        help='path where HCP data will be stored in BIDS format'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    hcp_to_bids()