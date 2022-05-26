#!/usr/bin/python

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import json
import webdataset as wds
from collections import OrderedDict
import nibabel as nb
from nilearn.image import resample_to_img
from nilearn.datasets import fetch_atlas_difumo
from nilearn.input_data import NiftiMapsMasker
from templateflow import api as tflow
from load_confounds import Confounds


class Preprocessor:
    def __init__(self,
        root: str,
        dataset: str,
        derivatives_path: str=None,
        verbose: bool=False,
        t_r: float=None
        ) -> None:

        if derivatives_path is None:
            derivatives_path = [
                'fmriprep-20.2.3/fmriprep/',
                'fmriprep-20.2.0/fmriprep/',
                'fmriprep/',
                'HCPpipelines/fMRIVolume/'
            ]
        self.root = root
        self.dataset = dataset
        self.verbose = verbose
        self._check_derivatives_path(derivatives_path)
        self.subjects = self.get_subjects()
        self.subjects.sort()
        self.preproc_bold_files = None
        self.source_bold_files = None
        self.t_r = t_r
        self.set_template()
        self.set_difumo()
        self.set_confounds_strategy()
        if self.verbose:
            print(
                "Found {} subjects with each ~{} preproc func files".format(
                    len(self.subjects), self.count_preproc_bold() // len(self.subjects))
            )

    def get_subjects(
        self,
        derivatives: str=None
        ) -> List[str]:
        target_dir = derivatives if derivatives is not None else self.derivatives_path
        return list(
            np.unique(
                [
                    f.split('sub-')[1] for f in os.listdir(target_dir)
                    if f.startswith('sub-')
                    and os.path.isdir(os.path.join(target_dir, f))
                    and 'html' not in f
                ]
            )
        )

    @staticmethod
    def _identify_bold_markers(
        filepath: str=None
        ) -> Tuple[str]:
        filename = filepath.split('/')[-1]
        markers = OrderedDict({'sub': filename.split('sub-')[1].split('_')[0]})
        for marker in ['ses', 'task', 'acq', 'ce', 'rec', 'dir', 'echo', 'part', 'run']:
            markers[marker] = filename.split('{}-'.format(marker))[1].split('_')[0] if '_{}-'.format(marker) in filepath else None
        return markers

    def _make_bold_filters(
        self,
        filepath: str
        ):
        filters = []
        for marker, marker_v in self._identify_bold_markers(filepath).items():
            if marker_v is not None:
                if marker == 'run':
                    try:
                        marker_v = int(marker_v)
                    except ValueError:
                        marker_v = str(marker_v)
                filters.append(
                    '{}-{}'.format(marker, marker_v)
                )
        return filters

    @staticmethod
    def _identify_func_subdirs(
        dir: str=None
        ) -> list:
        if os.path.isdir(os.path.join(dir, 'func')):
            return [os.path.join(dir, 'func')]
        subdirs = os.listdir(dir)
        if np.any([f.startswith('run-') for f in subdirs]):
            subdir_identifier = 'run'
        elif np.any([f.startswith('ses-') for f in subdirs]):
            subdir_identifier = 'ses'
        elif np.any([f.startswith('task-') for f in subdirs]):
            subdir_identifier = 'task'
        else:
            return []
        return [
            os.path.join(dir, f, 'func')
            for f in subdirs
            if f.startswith(subdir_identifier)
            and os.path.isdir(os.path.join(dir, f))
            and 'func' in os.listdir(os.path.join(dir, f))
        ]

    def _check_derivatives_path(
        self,
        derivatives_path: str=None
        ) -> None:
        """identify derivatives directory"""
        if isinstance(derivatives_path, str):
            self.derivatives_path = os.path.join(
                self.root,
                'derivatives',
                derivatives_path
            )
            if not os.path.isdir(self.derivatives_path):
                raise OSError(
                    'no derivatives found in {}.'.format(
                        self.derivatives_path
                    )
                )
        elif isinstance(derivatives_path, list):
            for deriv in derivatives_path:
                self.derivatives_path = os.path.join(
                    self.root,
                    'derivatives',
                    deriv
                )
                if not os.path.isdir(self.derivatives_path):
                    self.derivatives_path = None
                else:
                    break
            if self.derivatives_path is None:
                raise OSError(
                    'no {} subdir found in found in {}.'.format(
                        os.path.join(
                            self.root,
                            'derivatives'
                        ),
                        self.root
                    )
                )

    def _collect_preproc_bold_files(self):
        self.preproc_bold_files = {
            subject: self.get_subject_deriv_files(
                subject=subject,
                filters=[
                    'MNI152NLin2009cAsym',
                    'bold',
                    'preproc',
                    '.nii.gz'
                ]
            ) for subject in self.subjects
        }

    def _collect_source_bold_files(self):
        self.source_bold_files = {
            subject: self.get_subject_source_files(
                subject=subject,
                filters=[
                    'bold',
                    '.nii'
                ]
            ) for subject in self.subjects
        }

    def count_preproc_bold(self):
        if self.preproc_bold_files is None:
            self._collect_preproc_bold_files()
        return np.sum(
            [
                len(v) for v in self.preproc_bold_files.values()
                if v is not None
            ]
        )

    def count_source_bold(self):
        if self.source_bold_files is None:
            self._collect_source_bold_files()
        return np.sum(
            [
                len(v) for v in self.source_bold_files.values()
                if v is not None
            ]
        )
        
    def get_subject_deriv_files(
        self,
        subject,
        filters,
        ) -> List[str]:
        return self._get_subject_files(
            subject=subject,
            filters=filters,
            basedir=self.derivatives_path
        )

    def get_subject_source_files(
        self,
        subject,
        filters
        ) -> List[str]:
        return self._get_subject_files(
            subject=subject,
            filters=filters,
            basedir=self.root
        )

    def _get_subject_files(
        self,
        subject,
        filters,
        basedir
        ) -> List[str]:
        s_files = None
        if os.path.isdir(basedir):
            s_dir = os.path.join(
                basedir,
                'sub-{}'.format(subject)
            )
            if os.path.isdir(s_dir):
                s_subdirs = self._identify_func_subdirs(s_dir)
                s_files = []
                for s_subdir in s_subdirs:  
                    s_files += self._get_filepaths(
                        path=s_subdir,
                        filters=filters,
                    )
                if s_files:
                    s_files.sort()  
        return s_files

    def _get_filepaths(
        self,
        path,
        filters
        ) -> List[str]:
        return [
            os.path.join(path, p)
            for p in os.listdir(path)
            if all(p.find(f)!=-1 for f in list(filters))
        ]

    def set_template(
        self,
        resolution=2
        ) -> None:
        self.template = nb.load(
            tflow.get(
                'MNI152NLin2009cAsym',
                desc=None,
                resolution=resolution,
                suffix='T1w'
            )
        )

    def set_difumo(
        self,
        dimension=1024,
        resolution=2
        ) -> None:
        difumo_maps_path = os.path.join(
            Path.home(),
            'nilearn_data',
            'difumo_atlases',
            str(dimension),
            '{}mm'.format(resolution),
            'maps.nii.gz'
        )
        if os.path.isfile(difumo_maps_path):
            self.difumo = nb.load(difumo_maps_path)
        else:
            self.difumo = fetch_atlas_difumo(
                dimension=dimension,
                resolution_mm=resolution
            ).maps
            
    def set_masker(
        self,
        smoothing_fwhm: int=3,
        standardize: str='zscore',
        detrend: bool=True,
        verbose: int=0,
        memory_level: int=5,
        memory: str=None,
        maps_img = None,
        t_r: float=None,
        high_pass: float=0.008,
        ) -> NiftiMapsMasker:
        if memory is None:
            self._masker_cache = os.path.join(
                self.root,
                '.cache/'
            )
        else:
            self._masker_cache = memory
        os.makedirs(self._masker_cache, exist_ok=True)
        if maps_img is None:
            maps_img = self.difumo
        t_r = self.t_r if t_r is None else t_r
        high_pass = high_pass if t_r is not None else None
        self.masker = NiftiMapsMasker(
                maps_img=self.difumo,
                smoothing_fwhm=smoothing_fwhm,
                standardize=standardize,
                detrend=detrend,
                verbose=verbose,
                memory_level=memory_level,
                t_r=t_r,
                high_pass=high_pass,
                memory=self._masker_cache
            )

    def _clear_masker_cache(self):
        cache_files = os.listdir(self._masker_cache)
        for f in cache_files:
            if os.path.isfile(f):
                os.remove(
                    os.path.join(
                        self._masker_cache,
                        f
                    )
                )
            elif os.path.isdir(f):
                os.rmdir(
                    os.path.join(
                        self._masker_cache,
                        f
                    )
                )

    def set_confounds_strategy(
        self,
        strategy: List[str]=None,
        motion: str="basic",
        wm_csf: str="basic",
        demean: bool=True,
        global_signal: str="basic",
        **kwargs) -> None:
        if strategy is None:
            strategy = [
                'motion',
                'wm_csf',
                'global'
            ]
        self.confounds = Confounds(
            strategy=strategy,
            motion=motion,
            wm_csf=wm_csf,
            demean=demean,
            global_signal=global_signal,
            **kwargs
        )
        
    def make_key(self, bold_path):
        key = "ds-{}".format(
            self.dataset
        )
        for marker, marker_v in self._identify_bold_markers(bold_path).items():
            if marker_v is not None:
                if marker == 'run':
                    try:
                        marker_v = int(marker_v)
                    except ValueError:
                        marker_v = str(marker_v)
                key = '{}_{}-{}'.format(
                    key, marker, marker_v
                )
        return key

    @staticmethod
    def write_bold_to_tar(
        bold=None,
        path=None,
        key=None,
        sink=None,
        sample_dict=None,
        t_r=None,
        ) -> None:
        if sample_dict is None:
            assert isinstance(key, str), 'key needs to be a string'
            assert isinstance(bold, np.ndarray), 'bold needs to in numpy array format (shape: timepoints x networks)'
            sample_dict = {
                '__key__': key,
                'bold.pyd': bold.astype(np.float32)
            }
            if t_r is not None:
                assert isinstance(t_r, (int, float)), 't_r needs to be a number'
                sample_dict["t_r.pyd"] = np.float32(t_r)
        else:
            assert isinstance(sample_dict, dict), 'sample_dict needs to be dict'
            assert '__key__' in sample_dict.keys(), 'sample_dict needs to contain __key__ entry'
            assert 'bold.pyd' in sample_dict.keys(), 'sample_dict needs to contain bold.pyd entry'
            assert isinstance(sample_dict['__key__'], str), '__key__ entry needs to be a string.'
            assert isinstance(sample_dict['bold.pyd'], np.ndarray), 'bold.pyd entry needs to in numpy array format (shape: timepoints x networks)'
            sample_dict['bold.pyd'] = sample_dict['bold.pyd'].astype(np.float32)
            if 't_r.pyd' in sample_dict.keys():
                sample_dict['t_r.pyd'] = np.float32(sample_dict['t_r.pyd'])
        if sink is not None:
            assert isinstance(sink, wds.TarWriter), 'sink needs to be tarwriter from webdataset'
            sink.write(sample_dict)
        else:
            assert path is not None, 'path must be specified'
            os.makedirs(path, exist_ok=True)
            sink_path = '{}/{}.tar'.format(
                path, key
            )
            with wds.TarWriter(sink_path) as sink:
                sink.write(sample_dict)

    def write_all_preproc_bold_to_tar(
        self,
        path: str=None,
        t_r: float=None
        ) -> None:
        """write preproc bold files to .tar
        """
        os.makedirs(path, exist_ok=True)
        for sub in self.subjects:
            if self.verbose:
                print('\tWriting data for subject: {}'.format(sub))
            sub_bolds = self.get_subject_deriv_files(
                subject=sub,
                filters=[
                    'MNI152NLin2009cAsym',
                    'bold',
                    'preproc',
                    '.nii.gz'
                ]
            )
            for bold_path in sub_bolds:
                if t_r is None and self.t_r is None:
                    t_r = self.get_bold_tr(bold_path)
                preproc_bold = self.preprocess_bold(
                    bold_path=bold_path,
                    t_r=self.t_r if t_r is None else t_r
                )
                key = self.make_key(bold_path)
                if self.verbose:
                    print('\tWriting: {}'.format(bold_path))
                self.write_bold_to_tar(
                    bold=preproc_bold,
                    path=path,
                    key=key,
                    t_r=self.t_r if t_r is None else t_r 
                )
        if self.verbose:
            print('done.') 

    @staticmethod
    def get_bold_tr(bold_path):
        json_sidecar_path = '{}.json'.format(
            bold_path.split('.nii.gz')[0]
        )
        if os.path.isfile(json_sidecar_path):
            with open(json_sidecar_path, 'r') as jfile:
                jdata=jfile.read()
            t_r = json.loads(jdata)['RepetitionTime']
        else:
            t_r = nb.load(bold_path).header.get_zooms()[-1]
        return round(float(t_r), 5)

    def preprocess_bold(
        self, 
        bold_path: str=None,
        t_r: float=None):
        assert bold_path is not None, 'bold_path is required.'
        assert os.path.isfile(bold_path), 'no file found at {}'.format(bold_path)
        filters = self._make_bold_filters(bold_path)
        filters += [
            'confounds',
            '.tsv'
        ]
        sub = filters[0].split('sub-')[1]
        confounds_file = self.get_subject_deriv_files(
            subject=sub,
            filters=filters
        )
        confounds = self.confounds.load(bold_path) if len(confounds_file) > 0 else None
        if confounds is not None:
            if t_r is None and self.t_r is None:
                t_r = self.get_bold_tr(bold_path)
            self.set_masker(
                t_r=self.t_r if t_r is None else t_r,
                memory=os.path.join(
                    self.root,
                    'sub-{}'.format(sub),
                    '.cache/'
                )
            )
            imgs = nb.load(bold_path)
            imgs_resampled = resample_to_img(imgs, self.template)
            imgs_masked = self.masker.fit_transform(
                imgs=imgs_resampled,
                confounds=confounds
            )
            self._clear_masker_cache()
            return imgs_masked
        else:
            return None