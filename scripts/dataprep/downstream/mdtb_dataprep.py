#!/usr/bin/env python3

import sys, os
import argparse
from typing import Dict, Generator, Tuple
import json
import numpy as np
import pandas as pd
import webdataset as wds
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../../../')
from src.preprocessor import Preprocessor


def preprocess_mdtb(config: Dict=None) -> None:
    """Script's main function; additional preprocessing
    of MDTB fmriprep derivatives'"""

    if config is None:
        config = vars(get_args().parse_args())

    dataset = 'ds002105'
    tasks = ['a', 'b']
    runs = np.arange(1,9)
    sessions = [1, 2]
    t_r = 1
    t_r_offset = 3 * t_r
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
    os.makedirs(
        os.path.join(
            config["data_dir"],
            dataset
        ),
        exist_ok=True
    )
    task_label_mapping_path = os.path.join(
        config["data_dir"],
        dataset,
        'task_label_mapping.json'
    )

    if not os.path.isfile(task_label_mapping_path):
        print(
            '\tmapping tasks to labels ...'
        )
        task_label_mapping = map_tasks_to_labels(
            ds_layout=ds_layout,
            tasks=tasks,
            sessions=sessions,
            runs=runs
        )

        with open(task_label_mapping_path, 'w') as f:
            json.dump(task_label_mapping, f)

    else:
        print(
            f'\tfound existing task-label mapping at {task_label_mapping_path}'
        )
        with open(task_label_mapping_path, 'r') as f:
            task_label_mapping = json.load(f)

    for t, ti in task_label_mapping.items():
        print(
            f'\t\t{t} -> {ti}'
        )  

    print(
        '\nStarting preprocessing of bold data ...'
    )

    for task in tasks:
        
        for session in sessions:
            
            for run in runs:
                bold_path = ds_layout.get_subject_deriv_files(
                    subject=config["subject"],
                    filters=[
                        f"task-{task}",
                        f"ses-{task}{session}",
                        f"run-{run}",
                        "space-MNI152NLin2009cAsym",
                        "desc-preproc_bold",
                        'nii.gz'
                    ]
                )
                
                if bold_path is not None:
                    assert len(bold_path)==1, 'too many bold files found!'
                    bold_path = bold_path[0]
                    print(
                        f'\tpreprocessing: {bold_path}'
                    )
                    preproc_bold = ds_layout.preprocess_bold(
                        bold_path=bold_path,
                        t_r=t_r
                    )
                    trs = np.arange(preproc_bold.shape[0]) * t_r
                    key = ds_layout.make_key(bold_path)
                    sink_path = os.path.join(
                        config["data_dir"],
                        dataset,
                        f'{key}.tar'
                    )
                    
                    if os.path.isfile(sink_path):
                        print(
                            f'\t/!\ skipping sub-{config["subject"]}, '
                            f'task-{task}, '
                            f'ses-{session}, '
                            f'run-{run}, '
                            f'as {sink_path} exists already'
                        )
                        continue
                    
                    ev_path = ds_layout.get_subject_source_files(
                        subject=config["subject"],
                        filters=[
                            f"task-{task}",
                            f"ses-{task}{session}",
                            f"run-{run}",
                            'events.tsv'
                        ]
                    )

                    if len(ev_path)==1:
                        ev_iterator = yield_task_ev(ev_path=ev_path[0])
                        
                        if ev_iterator is not None:
                        
                            with wds.TarWriter(sink_path) as sink:
                            
                                for (
                                        sample_i,
                                        (
                                            ev_task,
                                            ev_on,
                                            ev_off
                                        )
                                    ) in enumerate(ev_iterator, start=1):
                                    ev_idx = np.logical_and(
                                        trs >= (ev_on+t_r_offset),
                                        trs < (ev_off+t_r_offset)
                                    )
                                    
                                    if np.sum(ev_idx) > 0:
                                        ev_bold = preproc_bold[ev_idx]
                                        ev_label_task = int(task_label_mapping[ev_task])
                                        sample_key = '{}_sample_{:03d}'.format(
                                            key,
                                            sample_i
                                        )
                                        sample_dict = {
                                            '__key__': sample_key,
                                            'bold.pyd': ev_bold.astype(np.float32),
                                            'task_label.pyd': np.int32(ev_label_task),
                                            'task_name': ev_task,
                                            't_r.pyd': np.float32(t_r)
                                        }
                                        sink.write(sample_dict)

    print('... done.')


def yield_task_ev(
    ev_path: str
    ) -> Generator[Tuple[str, float, float], None, None]:
    """Yield type, onset, and end of events in task EV file."""
    ev_df = pd.read_csv(ev_path, sep='\t')

    if ev_df.shape[0] < 1:
        return None

    out = []
    current_task = (
        ev_df['taskName'][0][:-1]
        if ev_df['taskName'][0].endswith('2')
        else ev_df['taskName'][0]
    )
    task_start = ev_df['onset'][0]

    for ev in ev_df.itertuples():

        if ev.taskName != current_task:

            if 'instruct' not in current_task:
                task_end = float(ev.onset)
                out.append(
                    (
                        current_task,
                        task_start,
                        task_end
                    )
                )
            task_start = float(ev.onset)
            current_task = (
                ev.taskName[:-1]
                if ev.taskName.endswith('2')
                else ev.taskName
            )

    if out:
        yield from out
    
    else:
        return None


def map_tasks_to_labels(
    ds_layout,
    tasks,
    sessions,
    runs
    ) -> Dict:
    """Map task names to numeric labels."""
    task_labels = []

    for subject in ds_layout.subjects:

        for task in tasks:

            for session in sessions:

                for run in runs:
                    ev_path = ds_layout.get_subject_source_files(
                        subject=subject,
                        filters=[
                            f"task-{task}",
                            f"ses-{task}{session}",
                            f"run-{run}",
                            'events.tsv'
                        ]
                    )

                    if len(ev_path) == 0:
                        print(
                            "/!\ no ev file found for "
                            f"task-{task}, "
                            f"ses-{task}{session}, "
                            f"run-{run} "
                        )
                        continue

                    assert len(ev_path)==1, 'this should be exactly 1 ev file!'
                    ev_path = ev_path[0]

                    if os.path.isfile(ev_path):
                        ev_df = pd.read_csv(ev_path, sep='\t')

                        for ev in ev_df.itertuples():
                            ev_task = (
                                ev.taskName[:-1]
                                if ev.taskName.endswith('2')
                                else ev.taskName
                            )

                            if ev_task not in task_labels and 'instruct' not in ev_task:
                                task_labels.append(ev_task)  

    return {t: ti for ti, t in enumerate(sorted(task_labels))}   


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='additional preprocessing of MDTB fmriprep derivatives'
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
        default='../data/downstream/',
        type=str,
        help='path where .tar files for fMRI runs wil be stored '
             '(default: ../data/downstream)'
    )
    parser.add_argument(
        '--subject',
        metavar='SUBJECT',
        type=str,
        help='id of subject for which data is preprocessed'
    )
    return parser


if __name__ == '__main__':
    preprocess_mdtb()