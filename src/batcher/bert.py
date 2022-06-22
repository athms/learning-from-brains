#!/usr/bin/env python3

from typing import Dict, Generator
import numpy as np
import torch
import webdataset as wds
from src.batcher.base import BaseBatcher, _pad_seq_right_to_n


class BERTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataloader_a,
        dataloader_b,
        length,
        seq_max,
        sample_keys = None,
        ) -> None:
        self.name = 'BERTDataset'
        self.dataloader_a = iter(dataloader_a)
        self.dataloader_b = iter(dataloader_b)
        self._length = length
        self.seq_max = seq_max
        self.sample_keys = sample_keys

    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):

        sample = self._combine_to_sample(
            sample_a=next(self.dataloader_a),
            sample_b=next(self.dataloader_b)
        )
        
        if self.sample_keys is not None:
            sample = {
                key: sample[key] 
                for key in self.sample_keys
                if key in sample
            }
        
        return sample

    def _combine_to_sample(
        self,
        sample_a: Dict[str, np.ndarray],
        sample_b: Dict[str, np.ndarray] 
        ) -> Dict[str, np.ndarray]:
        bold_is_next = self._combine_seqs_as_is_next(
            sample_a=sample_a,
            sample_b=sample_b,
            key='bold'
        )
        bold_not_next = self._combine_seqs_as_not_next(
            sample_a=sample_a,
            sample_b=sample_b,
            key='bold'
        )
        t_rs_is_next = self._combine_seqs_as_is_next(
            sample_a=sample_a,
            sample_b=sample_b,
            key='t_rs'
        )
        t_rs_not_next = self._combine_seqs_as_not_next(
            sample_a=sample_a,
            sample_b=sample_b,
            key='t_rs'
        )
        attention_mask_is_next = self._combine_seqs_as_is_next(
            sample_a=sample_a,
            sample_b=sample_b,
            key='attention_mask'
        )
        attention_mask_not_next = self._combine_seqs_as_not_next(
            sample_a=sample_a,
            sample_b=sample_b,
            key='attention_mask'
        )
        batch_size = bold_is_next.shape[0]
        bold_a_shape = sample_a['bold_a.pyd'].shape
        bold_b_shape = sample_a['bold_b.pyd'].shape
        token_type_ids = np.concatenate(
            [
                np.zeros(
                    (
                        batch_size,
                        bold_a_shape[1]
                    )
                ),
                np.ones(
                    (
                        batch_size,
                        bold_b_shape[1]
                    )
                )
            ],
            axis=1
        )
        is_next = torch.rand(size=(batch_size,))
        is_next = is_next.ge(0.5).to(torch.int)
        bold = self._order_seqs_by_is_next(
            seqs_next=bold_is_next,
            seqs_not_next=bold_not_next,
            is_next=is_next,
            batch_size=batch_size
        )
        t_rs = self._order_seqs_by_is_next(
            seqs_next=t_rs_is_next,
            seqs_not_next=t_rs_not_next,
            is_next=is_next,
            batch_size=batch_size
        )
        attention_mask = self._order_seqs_by_is_next(
            seqs_next=attention_mask_is_next,
            seqs_not_next=attention_mask_not_next,
            is_next=is_next,
            batch_size=batch_size
        )
        # remove padding between sequences
        attention_mask_bool = attention_mask.astype(np.bool)
        bold = [
            bold[i, attention_mask_bool[i]].reshape(
                -1,
                bold.shape[-1]
            )
            for i in range(batch_size)
        ]
        token_type_ids = [
            token_type_ids[i, attention_mask_bool[i]]
            for i in range(batch_size)
        ]
        t_rs = [
            t_rs[i, attention_mask_bool[i]]
            for i in range(batch_size)
        ]
        attention_mask = [
            np.ones(bold[i].shape[0])
            for i in range(batch_size)
        ]
        bold = self._pad_seqs_right_to_n(seqs=bold, n=self.seq_max)
        t_rs = self._pad_seqs_right_to_n(seqs=t_rs, n=self.seq_max)
        attention_mask = self._pad_seqs_right_to_n(seqs=attention_mask, n=self.seq_max)
        token_type_ids = self._pad_seqs_right_to_n(seqs=token_type_ids, n=self.seq_max)
        
        return  {
            'inputs': torch.from_numpy(bold).to(torch.float),
            't_rs': torch.from_numpy(t_rs).to(torch.float),
            'attention_mask': torch.from_numpy(attention_mask).to(torch.long),
            'token_type_ids': torch.from_numpy(token_type_ids).to(torch.long),
            'is_next': is_next.to(torch.long)
        }

    @staticmethod
    def _combine_seqs_as_is_next(
        sample_a: Dict[str,np.ndarray],
        sample_b: Dict[str,np.ndarray],
        key: str
        ) -> np.ndarray:
        return np.concatenate(
            [
                np.concatenate(
                    [
                        sample_a[f'{key}_a.pyd'],
                        sample_a[f'{key}_b.pyd']
                    ],
                    axis=1
                ),
                np.concatenate(
                    [
                        sample_b[f'{key}_a.pyd'],
                        sample_b[f'{key}_b.pyd']
                    ],
                    axis=1
                )
            ],
            axis=0
        )

    @staticmethod
    def _combine_seqs_as_not_next(
        sample_a: Dict[str, np.ndarray],
        sample_b: Dict[str, np.ndarray],
        key: str
        ) -> np.ndarray:
        return np.concatenate(
            [
                np.concatenate(
                    [
                        sample_a[f'{key}_a.pyd'],
                        sample_b[f'{key}_b.pyd']
                    ],
                    axis=1
                ),
                np.concatenate(
                    [
                        sample_b[f'{key}_a.pyd'],
                        sample_a[f'{key}_b.pyd']
                    ],
                    axis=1
                )
            ],
            axis=0
        )

    @staticmethod
    def _order_seqs_by_is_next(
        seqs_next: np.ndarray,
        seqs_not_next: np.ndarray,
        is_next: np.ndarray,
        batch_size: int = None
        ) -> np.ndarray:
        if batch_size is None:
            batch_size = seqs_next.shape[1]
        seqs = np.stack(
                [
                    seqs_next,
                    seqs_not_next
                ],
                axis=0
        )
        return np.stack(
            [
                seqs[is_next[i],i]
                for i in range(batch_size)
            ],
            axis=0
        )

    def _pad_seqs_right_to_n(
        self,
        seqs,
        n: int=None
        ) -> np.ndarray:
        if n is None:
            n = self.seq_max
        return np.stack(
            [
                _pad_seq_right_to_n(
                    seq=seq,
                    n=n
                )
                for seq in seqs
            ],
            axis=0
        )


class BERTBatcher(BaseBatcher):

    def __init__(
        self,
        gap_min: int = 1,
        gap_max: int = 5,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        assert gap_min >= 0, 'gap_min must be >= 0'
        assert gap_min <= gap_max, 'gap_min must be <= gap_max'
        assert gap_min < (self.seq_min - 1), 'gap_min must be < (seq_min - 1)'
        self.gap_min = gap_min
        self.gap_max = gap_max

    def _make_dataloader(
        self,
        files,
        repeat: bool = True,
        n_shuffle_shards: int = 1000,
        n_shuffle_samples: int = 1000,
        batch_size: int = 1,
        num_workers: int = 0
        ) -> torch.utils.data.DataLoader:
        dataset = wds.WebDataset(files)

        if n_shuffle_shards is not None:
            dataset = dataset.shuffle(n_shuffle_shards)

        dataset = dataset.decode("pil").map(self.preprocess_sample)

        if repeat:
            dataset = dataset.repeat()

        if n_shuffle_samples is not None:
            dataset = dataset.shuffle(n_shuffle_samples)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

    def dataset(
        self,
        tarfiles: list,
        repeat: bool = True,
        length: int = 400000,
        n_shuffle_shards: int = 1000,
        n_shuffle_samples: int = 1000,
        batch_size: int = 2,
        num_workers: int = 0
        ) -> torch.utils.data.Dataset:
        """Create Pytorch dataset that can be used for training.

        Args:
        -----
            tarfiles: list
                List of paths to data files (ie., fMRI runs) used for training.
            repeat: bool
                If True, repeat the dataset indefinitely.
            length: int
                Maximum number of samples to yield from the dataset.
            n_shuffle_shards: int
                Buffer for shuffling of tarfiles during training.
            n_shuffle_samples: int
                Buffer for shuffling of samples during training.
            batch_size: int
                Number of samples per batch; must be a multiple of 2!
            num_workers: int
                Number of workers to use for data loading.

        Returns:
        -----
            torch.utils.data.Dataset: Pytorch dataset.
        """
        assert len(tarfiles) >= 2, 'batcher requires more than one tarfile.'
        tarfiles_a = tarfiles[:int(len(tarfiles)/2)]
        tarfiles_b = tarfiles[int(len(tarfiles)/2):]
        assert set(tarfiles_a).isdisjoint(tarfiles_b), 'tarfiles_a and tarfiles_b are not disjoint.'
        assert batch_size > 0, 'batch_size must be > 0'
        assert batch_size%2==0, 'batch_size must be even'
        dataloader_a = self._make_dataloader(
            files=tarfiles_a,
            repeat=repeat,
            batch_size=int(batch_size/2),
            n_shuffle_shards=n_shuffle_shards,
            n_shuffle_samples=n_shuffle_samples,
            num_workers=num_workers
        )
        dataloader_b = self._make_dataloader(
            files=tarfiles_b,
            repeat=repeat,
            batch_size=int(batch_size/2),
            n_shuffle_shards=n_shuffle_shards,
            n_shuffle_samples=n_shuffle_samples,
            num_workers=num_workers
        )
        return BERTDataset(
            dataloader_a=dataloader_a,
            dataloader_b=dataloader_b,
            sample_keys=self.sample_keys,
            length=length,
            seq_max=self.seq_max
        )  

    def preprocess_sample(
        self,
        sample: Dict[str, np.ndarray]
        ) -> Dict[str, np.ndarray]:
        out = dict(__key__=sample["__key__"])
        t_r = sample["t_r.pyd"]
        for key, value in sample.items():
            
            if key == "bold.pyd":

                bold = np.array(value).astype(np.float)

                if self.bold_dummy_mode:
                    bold = self.make_bold_dummy(
                        bold_shape=bold.shape,
                        t_r=t_r
                    )

                bold_len = bold.shape[0]
                seq_on, seq_len = self._sample_seq_on_and_len(bold_len=bold_len)
                gap = 0

                if self.gap_min < self.gap_max: 
                    gap = np.random.randint(
                        low=self.gap_min,
                        high=self.gap_max, 
                        size=1
                    )[0]

                # we want at least two t_rs before and after gap
                if gap > (seq_len - 4):
                    gap = 0

                seq_split = int((seq_len - gap) / 2)
                bold_a = bold[seq_on:seq_on+seq_split]
                t_rs_a = np.arange(bold_a.shape[0]) * t_r
                attention_mask_a = np.ones(bold_a.shape[0])
                out["bold_a.pyd"] = self._pad_seq_right_to_n(
                    seq=bold_a,
                    n=self.seq_max
                )
                out["t_rs_a.pyd"] = self._pad_seq_right_to_n(
                    seq=t_rs_a,
                    n=self.seq_max
                )
                out["attention_mask_a.pyd"] = self._pad_seq_right_to_n(
                    seq=attention_mask_a,
                    n=self.seq_max
                )
                bold_b = bold[seq_on+seq_split+gap:seq_on+seq_len]
                t_rs_b = np.arange(bold_b.shape[0]) * t_r
                attention_mask_b = np.ones(bold_b.shape[0])
                out["bold_b.pyd"] = self._pad_seq_right_to_n(
                    seq=bold_b,
                    n=self.seq_max
                )
                out["t_rs_b.pyd"] = self._pad_seq_right_to_n(
                    seq=t_rs_b,
                    n=self.seq_max
                )
                out["attention_mask_b.pyd"] = self._pad_seq_right_to_n(
                    seq=attention_mask_b,
                    n=self.seq_max
                )
                out['seq_on'] = seq_on
                out['gap'] = gap
                out['seq_len'] = seq_len

            elif key in {
                f"{self.decoding_target}.pyd",
                self.decoding_target
                }:
                out["labels"] = value

            else:
                out[key] = value

        return out