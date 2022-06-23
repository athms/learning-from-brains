#!/usr/bin/env python3

import os, sys
import argparse
from typing import Dict
import numpy as np
import nibabel as nb
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../../../')
from src.preprocessor import Preprocessor


def dataprep(config: Dict=None) -> None:
    """Script's main function; additional preprocessing 
    of fmriprep derivatives with Preprocessor from src/"""

    if config is None:
        config = vars(get_args().parse_args())

    if not os.path.isdir(
        os.path.join(
            config['bids_dir'],
            config['ds']
        )
    ):
        raise OSError(
            f'{config["ds"]} not found in {config["bids_dir"]}'
        )

    ds_root = os.path.join(
        config['bids_dir'],
        config['ds'],
        'data'
    )
    ds_out = os.path.join(
        config['data_dir'],
        config['ds']
    )
    preprocessor = Preprocessor(
        root=ds_root,
        dataset=config['ds'],
        verbose=False,
        t_r=float(config['tr']) if config['tr']!=-1 else None 
    )    
    assert config['subject'] in preprocessor.subjects,\
        f'sub-{config["subject"]} not found in {config["ds"]}.'
    print(
        f'Preprocessing data of sub-{config["subject"]}'
    )

    if 'fmriprep-20.2.0' in preprocessor.derivatives_path:
        # should data be tested for fmriprep 20.2.0 bug?
        # see: https://github.com/nipreps/fmriprep/issues/2307
        check_fmriprep_bug = config['check_fmriprep_bug'] == 'True'

    else:
        check_fmriprep_bug = False

    if not check_fmriprep_bug:
        print(
            '! Data not tested for fmriprep-20.2.0 bug'
        )
    
    def deriv_bold_iterator():
        """iterate BOLD derivatives that are to be preprocessed."""
        
        if check_fmriprep_bug:
            # make sure that files are not affected by fmriprep-20.2.0 bug
            # see: https://github.com/nipreps/fmriprep/issues/2307
            source_bolds = preprocessor.get_subject_source_files(
                subject=config['subject'],
                filters=[
                    'bold',
                    '.nii'
                ]
            )

            for source_bold in source_bolds:
                affected, _ = quality_check(nb.load(source_bold))

                if affected:
                    print(
                        f'skipping {source_bold}, because source data affected by fmriprep-20.2.0 bug.'
                    )

                else:
                    filters = preprocessor._make_bold_filters(source_bold)
                    filters += [
                        'space-MNI152NLin2009cAsym',
                        'preproc',
                        'bold',
                        'nii.gz'
                    ]
                    deriv_bold = preprocessor.get_subject_deriv_files(
                        subject=config['subject'],
                        filters=filters
                    )

                    if len(deriv_bold)==1:
                        yield deriv_bold[0]

                    else:
                        print(
                            f'\tskipping {source_bold}, as no corresponding derivative found.'
                        )
        
        else:
            filters = [
                'space-MNI152NLin2009cAsym',
                'preproc',
                'bold',
                'nii.gz'
            ]

            yield from preprocessor.get_subject_deriv_files(
                subject=config['subject'],
                filters=filters
            )
    
    for deriv_bold in deriv_bold_iterator():
        key = preprocessor.make_key(deriv_bold)
        sink_path = '{}/{}.tar'.format(
            ds_out, key
        )

        if not os.path.isfile(sink_path) or os.path.getsize(sink_path) < 50000:
            print('\tpreprocessing {}'.format(deriv_bold))
            bold_t_r = preprocessor.get_bold_tr(deriv_bold) if config['tr']==-1 else config['tr']
            preproc_bold = preprocessor.preprocess_bold(
                bold_path=deriv_bold,
                t_r=bold_t_r
            )
            preprocessor.write_bold_to_tar(
                bold=preproc_bold,
                t_r=bold_t_r,
                key=key,
                path=ds_out
            )                    
            print('\t..done.')

        else:
            print('skipping {}, as {} exists already'.format(deriv_bold, sink_path))


def quality_check(img):
    """test of source BOLD data for fmriprep-20.2.0 bug.
    see: https://github.com/nipreps/fmriprep/issues/2307"""
    zooms = np.array([img.header.get_zooms()[:3]])
    A = img.affine[:3, :3]
    cosines = A / zooms
    diff = A - cosines * zooms.T

    return not np.allclose(diff, 0), np.max(np.abs(diff))
    

def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='additional preprocessing of fmriprep derivatives'
    )
    
    parser.add_argument(
        '--bids-dir',
        metavar='DIR',
        type=str,
        help='path to data directory, '
             'where source data and derivatives are stored '
             'for dataset in BIDS format'
    )
    parser.add_argument(
        '--ds',
        metavar='STR',
        type=str,
        help='ID of dataset that will be preprocessed'
    )
    parser.add_argument(
        '--subject',
        metavar='STR',
        type=str,
        help='ID of subject whose data will be preprocessed'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        default='../data/tarfiles',
        type=str,
        help='path where .tar files for fMRI runs are stored '
             '(default: ../data/tarfiles)'
    )
    parser.add_argument(
        '--tr',
        metavar='TR',
        default=-1,
        type=float,
        help='repetition time / TR of BOLD data (in seconds); '
             'will be infered from data files, if not set (or set to -1).'
    )
    parser.add_argument(
        '--check-fmriprep-bug',
        metavar='BOOL',
        default='True',
        choices=('True', 'False'),
        type=str,
        help='whether or not to test for fmriprep 20.2.0 bug '
             '(default: True)'
    )

    return parser

        
if __name__ == '__main__':
    dataprep()