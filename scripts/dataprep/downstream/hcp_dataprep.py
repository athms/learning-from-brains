#!/usr/bin/env python3

import os, sys
import argparse
from typing import Tuple, Generator
import numpy as np
import pandas as pd
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../../../')
from src.preprocessor import Preprocessor
import webdataset as wds



def preprocess_hcp(args: argparse.Namespace=None) -> None:

    if args is None:
        args = get_args()
    
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
    cognitive_states = {
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
    cognitive_state_label_mapping = {
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
        args.data_dir,
        dataset,
        'data'
    )
    ds_layout = Preprocessor(
        ds_root,
        dataset,
        verbose=True
    )
    assert args.sub in ds_layout.get_subjects(), f'sub-{args.sub} not found in {ds_root}'

    for task_i, task in enumerate(tasks):

        if 'REST' not in task:
            continue

        #for run_i, _ in enumerate(runs, start=1):
        run_i = np.random.choice(np.array([1,2]))

        bold_paths = ds_layout.get_subject_deriv_files(
            subject=args.sub,
            filters=[
                f"task-{task}",
                f"run-{run_i}",
                "space-MNI152NLin2009cAsym",
                "desc-preproc_bold",
                'nii.gz'
            ]
        )

        if bold_paths is not None:
            assert len(bold_paths)==1, \
                f'there should only be one bold file but {len(bold_paths)} found!'
            bold_path = bold_paths[0]
            preproc_bold = ds_layout.preprocess_bold(bold_path=bold_path)
            t_rs = np.arange(preproc_bold.shape[0]) * t_r
            key = ds_layout.make_key(bold_path)
            sink_path = os.path.join(
                args.tarfiles_dir,
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
                    args.tarfiles_dir,
                    dataset
                ),
                exist_ok=True
            )
            sink = wds.TarWriter(sink_path)

            ev_iterator = None
            if 'REST' not in task:
                ev_paths = ds_layout.get_subject_source_files(
                    subject=args.sub,
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
                    seq_max=50
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
                        ev_label_in_task = np.where(cognitive_states[task]==ev_type)[0][0]
                        ev_label_across_tasks = cognitive_state_label_mapping[task][ev_label_in_task]
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
    seq_min: int = 10,
    seq_max: int = 25
    ) -> Generator[Tuple[str, float, float], None, None]:
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
    parser = argparse.ArgumentParser(description='HCP data preprocessing')
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        type=str,
        help='path to directory where HCP data are stored '
    )
    parser.add_argument(
        '--tarfiles-dir',
        metavar='DIR',
        default='../data/tarfiles/downstream/',
        type=str,
        help='path where .tar shards will be stored '
             '(default: ../data/tarfiles/downstream/)'
    )
    parser.add_argument(
        '--sub',
        metavar='SUBJECT',
        type=str,
        help='id of subject whose data are preprocessed'
    )
    return parser.parse_args()


if __name__ == '__main__':

    preprocess_hcp()