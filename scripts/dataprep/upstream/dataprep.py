#!/usr/bin/python

import os
import argparse
import numpy as np
import nibabel as nb
import nit



def dataprep(args: argparse.Namespace=None) -> None:

    # get parser args
    if args is None:
        args = get_args()

    # check whether dataset exists
    if not os.path.isdir(os.path.join(args.data, args.ds)):
        raise OSError('{} not found in {}'.format(args.ds, args.data))

    ds_root = os.path.join(args.data, args.ds, 'data')
    ds_out = os.path.join(args.out, args.ds)

    nit_preprocessor = nit.Preprocessor(
        root=ds_root,
        dataset=args.ds,
        verbose=False,
        t_r=float(args.tr) if args.tr!=-1 else None 
    )    
    assert args.sub in nit_preprocessor.subjects, 'sub-{} not found in {}.'.format(args.sub, args.ds)
    print('Preprocessing data of sub-{}'.format(args.sub))

    if 'fmriprep-20.2.0' in nit_preprocessor.derivatives_path:
        check_fmriprep_bug = args.check_fmriprep_bug == 'True'
    else:
        check_fmriprep_bug = False
    if not check_fmriprep_bug:
        print('Testing for fmriprep-20.2.0 bug is turned off')
    
    def deriv_bold_iterator():
        if check_fmriprep_bug:
            # make sure that files are not affected by fmriprep-20.2.0 bug!
            source_bolds = nit_preprocessor.get_subject_source_files(
                subject=args.sub,
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
                    filters = nit_preprocessor._make_bold_filters(source_bold)
                    filters += [
                        'space-MNI152NLin2009cAsym',
                        'preproc',
                        'bold',
                        'nii.gz'
                    ]
                    deriv_bold = nit_preprocessor.get_subject_deriv_files(
                        subject=args.sub,
                        filters=filters
                    )
                    if len(deriv_bold)==1:
                        yield deriv_bold[0]
                    else:
                        print('\tskipping {}, as no unique matching deriv bold file found.'.format(source_bold))
        else:
            filters = [
                'space-MNI152NLin2009cAsym',
                'preproc',
                'bold',
                'nii.gz'
            ]
            yield from nit_preprocessor.get_subject_deriv_files(
                subject=args.sub,
                filters=filters
            )
    
    for deriv_bold in deriv_bold_iterator():
        key = nit_preprocessor.make_key(deriv_bold)
        sink_path = '{}/{}.tar'.format(
            ds_out, key
        )
        if not os.path.isfile(sink_path) or os.path.getsize(sink_path) < 50000:
            print('\tpreprocessing {}'.format(deriv_bold))
            bold_t_r = nit_preprocessor.get_bold_tr(deriv_bold) if args.tr==-1 else args.tr
            preproc_bold = nit_preprocessor.preprocess_bold(
                bold_path=deriv_bold,
                t_r=bold_t_r
            )
            nit_preprocessor.write_bold_to_tar(
                bold=preproc_bold,
                t_r=bold_t_r,
                key=key,
                path=ds_out
            )                    
            print('\t..done.')
        else:
            print('skipping {}, as {} exists already'.format(deriv_bold, sink_path))


def quality_check(img):
    zooms = np.array([img.header.get_zooms()[:3]])
    A = img.affine[:3, :3]
    
    cosines = A / zooms
    diff = A - cosines * zooms.T

    return not np.allclose(diff, 0), np.max(np.abs(diff))
    

def get_args() -> argparse.Namespace:
    # parse input arguments
    parser = argparse.ArgumentParser(description='data preprocessing')
    
    parser.add_argument(
        '--data',
        metavar='DIR',
        type=str,
        help='path to root BIDS directory '
    )
    parser.add_argument(
        '--ds',
        metavar='STR',
        type=str,
        help='ID of dataset in data/'
    )
    parser.add_argument(
        '--sub',
        metavar='STR',
        type=str,
        help='ID of subject whose data will be preprocessed'
    )
    parser.add_argument(
        '--out',
        metavar='DIR',
        default='../data/tarfiles/',
        type=str,
        help='path where .tar shards are stored '
             '(default: ../data/tarfiles/)'
    )
    parser.add_argument(
        '--tr',
        metavar='TR',
        default=-1,
        type=float,
        help='TR of func data '
             'will be infered from data if not set.'
    )
    parser.add_argument(
        '--check-fmriprep-bug',
        metavar='BOOL',
        default='True',
        choices=('True', 'False'),
        type=str,
        help='whether or not to test for fmriprep-bug '
             '(default: True)'
    )

    return parser.parse_args()

        
if __name__ == '__main__':

    dataprep()