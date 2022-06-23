#!/usr/bin/env python3

import os, sys
import argparse
from typing import Tuple, Generator, Dict
import numpy as np
import pandas as pd
import webdataset as wds
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../../../')
from src.preprocessor import Preprocessor


def preprocess_hcp(config: Dict=None) -> None:
    """Script's main function; additional preprocessing
    of HCP fmriprep derivatives'"""

    if config is None:
        config = vars(get_args().parse_args())

    dataset = 'HCP'
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
    mental_states = {
        'EMOTION': np.array(
            [
                'fear',
                'neut'
            ]
        ),
        'GAMBLING': np.array(
            [
                'loss',
                'win'
            ]
        ),
        'LANGUAGE': np.array(
            [
                'math',
                'story'
            ]
        ),
        'MOTOR': np.array(
            [
                'lf',
                'lh',
                'rf',
                'rh',
                't'
            ]
        ),
        'RELATIONAL': np.array(
            [
                'match',
                'relation'
            ]
        ),
        'SOCIAL': np.array(
            [
                'mental',
                'rnd'
            ]
        ),
        'WM': np.array(
            [
                'body',
                'faces',
                'places',
                'tools'
            ]
        ),
        'REST1': np.array(['rest']),
        'REST2': np.array(['rest'])
    }
    mental_state_label_mapping = {
        'EMOTION': np.array([0, 1]),
        'GAMBLING': np.array([2, 3]),
        'LANGUAGE': np.array([4, 5]),
        'MOTOR': np.array([6, 7, 8, 9, 10]),
        'RELATIONAL': np.array([11, 12]),
        'SOCIAL': np.array([13, 14]),
        'WM': np.array([15, 16, 17, 18]),
        'REST1': np.array([19]),
        'REST2': np.array([19])
    }
    t_r = 0.72
    t_r_offset = 4 * t_r
    ds_root = os.path.join(
        config["bids_dir"],
        dataset,
        'data'
    )
    ds_layout = Preprocessor(
        ds_root,
        dataset,
        verbose=True
    )
    assert config["subject"] in ds_layout.get_subjects(),\
        f'sub-{config["subject"]} not found in {ds_root}'

    for task_i, task in enumerate(tasks):

        for run_i, _ in enumerate(runs, start=1):
            bold_path = ds_layout.get_subject_deriv_files(
                subject=config["subject"],
                filters=[
                    f"task-{task}",
                    f"run-{run_i}",
                    "space-MNI152NLin2009cAsym",
                    "desc-preproc_bold",
                    'nii.gz'
                ]
            )

            if bold_path is not None:
                assert len(bold_path)==1, \
                    f'there should only be one bold file but {len(bold_path)} found!'
                bold_path = bold_path[0]
                preproc_bold = ds_layout.preprocess_bold(bold_path=bold_path)
                t_rs = np.arange(preproc_bold.shape[0]) * t_r
                key = ds_layout.make_key(bold_path)
                sink_path = os.path.join(
                    config["data_dir"],
                    dataset,
                    f'{key}.tar'
                )

                if os.path.isfile(sink_path):
                    print(
                        f'Skipping {key}, as {sink_path} exists already'
                    )
                    continue

                os.makedirs(
                    os.path.join(
                        config["data_dir"],
                        dataset
                    ),
                    exist_ok=True
                )
                sink = wds.TarWriter(sink_path)
                ev_iterator = None

                if 'REST' not in task:
                    ev_paths = ds_layout.get_subject_source_files(
                        subject=config["subject"],
                        filters=[
                            f"task-{task}",
                            f"run-{run_i}",
                            'EV.csv'
                        ]
                    )

                    if ev_paths is not None:
                        assert len(ev_paths)==1, 'more than one EV file found.'
                        ev_iterator = yield_task_ev(ev_paths[0])

                else:
                    ev_iterator = yiel_rest_ev(
                        t_rs=t_rs,
                        seq_min=10,
                        seq_max=30
                    )

                if ev_iterator is not None:

                    for (
                            sample_i,
                            (
                                ev_type,
                                ev_on,
                                ev_off
                            )
                        ) in enumerate(
                            ev_iterator,
                            start=1
                        ):
                        t_r_idx = np.logical_and(
                            t_rs >= (ev_on+t_r_offset),
                            t_rs < (ev_off+t_r_offset)
                        )

                        if np.sum(t_r_idx) > 0:
                            ev_bold = preproc_bold[t_r_idx]
                            task_label = task_i if 'REST' not in task else len(tasks)-1
                            ev_label_in_task = np.where(mental_states[task]==ev_type)[0][0]
                            ev_label_across_tasks = mental_state_label_mapping[task][ev_label_in_task]
                            sample_key = '{}_sample_{:03d}'.format(
                                    key,
                                    sample_i
                            )
                            sample_dict = {
                                '__key__': sample_key,
                                'bold.pyd': ev_bold.astype(np.float32),
                                'task_label.pyd': int(task_label),
                                'task_name': task,
                                'mental_state': ev_type,
                                'label_in_task.pyd': int(ev_label_in_task),
                                'label_across_tasks.pyd': int(ev_label_across_tasks),
                                't_r.pyd': np.float32(t_r)
                            }
                            ds_layout.write_bold_to_tar(
                                sink=sink,
                                sample_dict=sample_dict
                            )

def yield_task_ev(
    ev_path: str
    ) -> Generator[Tuple[str, float, float], None, None]:
    """Yield type, onset, and end of events in task EV file."""
    ev_df = pd.read_csv(ev_path)
    
    if ev_df.shape[0] < 1:
        return None
    
    for ev in ev_df.itertuples():
        yield (
            ev.event_type,
            ev.onset,
            ev.end
        )                            

def yiel_rest_ev(
    t_rs: np.array,
    seq_min: int = 10, # in seconds
    seq_max: int = 30  
    ) -> Generator[Tuple[str, float, float], None, None]:
    """Yield random intervals (between seq_min and seq_max) for resting state data."""
    out = []
    ev_on = 0

    while ev_on <= max(t_rs):
        seq_len = np.random.randint(
            low=seq_min,
            high=seq_max,
            size=1
        )[0]
        ev_off = ev_on + seq_len
        out.append(
            (
                'rest',
                ev_on,
                ev_off
            )
        )
        ev_on = ev_off

    if out > 0:
        yield from out
    
    else:
        return None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='additional preprocessing of HCP fmriprep derivatives'
    )
    parser.add_argument(
        '--bids-dir',
        metavar='DIR',
        type=str,
        help='directory where HCP source data and derivatives '
             'are stored in BIDS format'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        default='../data/downstream',
        type=str,
        help='directory where .tar files for fMRI runs will be stored '
             '(default: ../data/downstream)'
    )
    parser.add_argument(
        '--subject',
        metavar='SUBJECT',
        type=str,
        help='id of subject whose data are preprocessed'
    )
    return parser


if __name__ == '__main__':
    preprocess_hcp()