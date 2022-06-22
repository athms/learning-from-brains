#!/usr/bin/env python3

from typing import Dict
import warnings
import torch


class LinearBaseline(torch.nn.Module):
    def __init__(
        self,
        n_networks: int=1024,
        n_timesteps: int=50,
        num_decoding_classes: int=1,
        **kwargs
        ) -> None:
        super().__init__()
        self.name = 'LinearBaseline'
        self.is_decoding_mode = True
        self.n_networks = n_networks
        self.n_timesteps = n_timesteps
        self.num_decoding_classes = num_decoding_classes
        self.decoding_head = None
        self.add_decoding_head(self.num_decoding_classes)
        self.timseries_model = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.n_timesteps,
                out_features=1
            )
        )

    def switch_decoding_mode(
        self,
        is_decoding_mode: bool=True,
        num_decoding_classes: int=None
        ) -> None:
        self.is_decoding_mode = is_decoding_mode
        if self.is_decoding_mode:
            self.add_decoding_head(num_decoding_classes=num_decoding_classes)
        else:
            self.decoding_head = None

    def add_decoding_head(
        self,
        num_decoding_classes: int
        ) -> None:
        if self.decoding_head is not None:
            if self.num_decoding_classes == num_decoding_classes:
                warnings.warn(
                    'Warning: not overwriting decoding head, as '
                    f'{num_decoding_classes}-class decoding head exists.'
                )
                return None
            else:
                warnings.warn(
                    f'Warning: overwriting existing {num_decoding_classes}-class decoding head.'
                )
        self.num_decoding_classes = num_decoding_classes
        self.decoding_head = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.n_networks,
                out_features=self.num_decoding_classes
            )
        )
        return None
    
    def decode(
        self,
        outputs: Dict[str, torch.tensor],
        ) -> Dict[str, torch.tensor]:
        return outputs

    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        aggregated_timeseries = self.timseries_model(
            torch.transpose(batch['inputs_embeds'], 1, 2)
        )
        
        return {
            'decoding_logits': self.decoding_head(aggregated_timeseries[:,:,0])
        }