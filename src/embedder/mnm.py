
#/usr/bin/env python3

from typing import Dict
import torch
from src.embedder.csm import CSMEmbedder


class MNMEmbedder(CSMEmbedder):
    
    def __init__(
        self,
        masking_rate: float = 0.2,
        n_positions: int=512,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = 'MNMEmbedder'
        self.training_style = 'MNM'
        assert self.training_style in {'MNM', 'decoding'}, f'{self.training_style} not supported'
        self._root_training_style = 'MNM'
        self.masking_rate = masking_rate
        self.n_positions = n_positions
        self.msk_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, self.n_positions, 1)
            )
        )
        self.cls_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, 1, self.in_dim)
            )
        )
        self._embeds = [
            self.msk_embed,
            self.cls_embed,
        ]
        self._init_embeds()

    def mask_inputs(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        assert inputs_key in batch, f'{inputs_key} not found in batch'
        input_shape = batch[inputs_key].size()
        modelling_mask = torch.rand(
            size=(
                input_shape[0],
                1,
                self.in_dim
            ),
            device=batch[inputs_key].device
        )
        modelling_mask = modelling_mask.ge(1-self.masking_rate)
        modelling_mask = modelling_mask.repeat(
            1,
            input_shape[1],
            1
        )
        attention_mask_expanded = torch.unsqueeze(
            batch['attention_mask'],
            dim=2
        ).repeat(
            1,
            1,
            self.in_dim
        )
        batch['modelling_mask'] = torch.where(
            attention_mask_expanded == 1,
            modelling_mask,
            torch.zeros_like(modelling_mask)
        ).to(torch.long)
        batch['masked_inputs'] = torch.masked_select(
            input=batch[inputs_key],
            mask=batch['modelling_mask'].to(torch.bool)
        ).detach().clone()
        batch['inputs_embeds'] = torch.where(
            batch['modelling_mask'] == 1,
            self.msk_embed[:, :input_shape[1]].repeat(
                batch[inputs_key].size()[0],
                1,
                self.in_dim
            ),
            batch[inputs_key].to(torch.float)
        )

        return batch