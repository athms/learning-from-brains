#!/usr/bin/env python3

from typing import Dict, Tuple, Generator
import numpy as np
from typing import Dict
import webdataset as wds
import torch


def _pad_seq_right_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n-seq.shape[0],
                    *seq.shape[1:]
                )
            ) * pad_value,  
        ],
        axis=0,
    )


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataloader,
        length,
        sample_keys
        ) -> None:
        self.name = 'BaseDataset'
        self._length = length
        self.dataloader = iter(dataloader)
        self.sample_keys = sample_keys

    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        
        sample = next(self.dataloader)
            
        if self.sample_keys is not None:
            sample = {
                key: sample[key] 
                for key in self.sample_keys
                if key in sample
            }
        
        return sample


class BaseBatcher:

    def __init__(
        self,
        sample_random_seq: bool = True,
        seq_min: int = 10,
        seq_max: int = 50,
        sample_keys: Tuple[str] = None,
        decoding_target: str = None,
        seed: int =  None,
        bold_dummy_mode: bool = False,
        **kwargs
        ) -> None:
        assert seq_min > 0, "seq_min must be greater than 0"
        assert seq_min < seq_max, 'seq_min must be less than seq_max'
        self.sample_random_seq = sample_random_seq
        self.seq_min = seq_min
        self.seq_max = seq_max
        self.decoding_target = decoding_target
        self.sample_keys = sample_keys
        self.bold_dummy_mode = bold_dummy_mode
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
    def _make_dataloader(
        self,
        files,
        repeat: bool = True,
        n_shuffle_shards: int = 1000,
        n_shuffle_samples: int = 1000,
        batch_size: int = 1,
        num_workers: int = 0
        ) -> Generator[Dict[str, torch.tensor], None, None]:
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
        repeat: bool=True,
        length: int = 400000,
        n_shuffle_shards: int = 1000,
        n_shuffle_samples: int = 1000,
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
            num_workers: int
                Number of workers to use for data loading.

        Returns:
        -----
            torch.utils.data.Dataset: Pytorch dataset.
        """
        dataloader = self._make_dataloader(
            files=tarfiles,
            repeat=repeat,
            n_shuffle_shards=n_shuffle_shards,
            n_shuffle_samples=n_shuffle_samples,
            num_workers=num_workers
        )
        return BaseDataset(
            dataloader=dataloader,
            length=length,
            sample_keys=self.sample_keys
        )
        
    @staticmethod
    def _pad_seq_right_to_n(
        seq: np.ndarray,
        n: int,
        pad_value: float = 0
        ) -> np.ndarray:
        return _pad_seq_right_to_n(
            seq=seq,
            n=n,
            pad_value=pad_value
        )
    
    def _sample_seq_on_and_len(
        self,
        bold_len: int
        ) -> Tuple[int, int]:
        
        seq_on = 0
        if self.sample_random_seq and self.seq_min < bold_len:            
            seq_min = min(int(self.seq_min), bold_len)
            seq_max = min(int(self.seq_max), bold_len)

            if seq_min < seq_max:
                seq_len = np.random.randint(
                    low=seq_min,
                    high=seq_max,
                    size=1
                )[0]
            
            else:
                seq_len = seq_max
            
            if seq_len < bold_len:
                seq_on = np.random.choice(
                    np.arange(
                        0,
                        bold_len-seq_len,
                        seq_len
                    ),
                    size=1
                )[0]
        
        elif not self.sample_random_seq and self.seq_max < bold_len:
            seq_len = self.seq_max

        else:
            seq_len = bold_len

        return seq_on, seq_len

    def make_bold_dummy(
        self,
        bold_shape: Tuple[int, int],
        t_r: float, # in seconds
        f_s: Tuple[float]=None, # sine frequencies in seconds
        ) -> np.ndarray:
        f_s = np.array([4, 8, 12]) if f_s is None else np.array(f_s).flatten()
        np.random.shuffle(f_s)
        f = np.zeros((1, bold_shape[-1]))
        for i, f_i in enumerate(f_s):
            f[:,i::len(f_s)] = f_i
        t_offsets = np.random.choice(
            a=np.arange(0,3), # in TRs
            size=bold_shape[1],
            replace=True
        )
        t = np.concatenate(
            [
                np.arange(
                    t_offsets[i],
                    bold_shape[0]+t_offsets[i]
                ).reshape(-1,1)
                for i in range(bold_shape[1])
            ],
            axis=-1
        ) * t_r
        return np.sin((1. / f) * t)

    def preprocess_sample(
        self,
        sample
        ) -> Dict[str, torch.Tensor]:
        out = dict(__key__=sample["__key__"])
        t_r = sample["t_r.pyd"]

        label = None
        f_s = None
        if self.bold_dummy_mode and self.decoding_target is not None:
            label = np.random.choice([0, 1])
            f_s = np.array([1, 2, 4]) if label == 0 else np.array([6, 8, 10])                

        for key, value in sample.items():
            if key == "bold.pyd":

                bold = np.array(value).astype(np.float)
                
                if self.bold_dummy_mode:
                    bold = self.make_bold_dummy(
                        bold_shape=bold.shape,
                        t_r=t_r,
                        f_s=f_s
                    )

                seq_on, seq_len = self._sample_seq_on_and_len(bold_len=len(bold))
                bold = bold[seq_on:seq_on+seq_len]
                t_rs = np.arange(seq_len) * t_r
                attention_mask = np.ones(seq_len)
                bold = self._pad_seq_right_to_n(
                    seq=bold,
                    n=self.seq_max,
                    pad_value=0
                )
                t_rs = self._pad_seq_right_to_n(
                    seq=t_rs,
                    n=self.seq_max,
                    pad_value=0
                )
                attention_mask = self._pad_seq_right_to_n(
                    seq=attention_mask,
                    n=self.seq_max,
                    pad_value=0
                )
                out["inputs"] = torch.from_numpy(bold).to(torch.float)
                out['t_rs'] = torch.from_numpy(t_rs).to(torch.float)
                out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
                out['seq_on'] = seq_on
                out['seq_len'] = seq_len

            elif key in {
                f"{self.decoding_target}.pyd",
                self.decoding_target
                }:
                out["labels"] = value

            else:
                out[key] = value
        
        if self.sample_keys is not None:
            out = {
                key: out[key] 
                for key in self.sample_keys
                if key in out
            }

        if label is not None:
            out['labels'] = label
        
        return out