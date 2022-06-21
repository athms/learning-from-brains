
#/usr/bin/env python3

from typing import Dict
import torch
from src.embedder.csm import CSMEmbedder


class MSMEmbedder(CSMEmbedder):
    
    def __init__(
        self,
        masking_rate: float = 0.2,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = 'MSMEmbedder'
        self.training_style = 'MSM'
        assert self.training_style in {'MSM', 'decoding'}, f'{self.training_style} not supported'
        self._root_training_style = 'MSM'
        self.masking_rate = masking_rate

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
                input_shape[1]
            ),
            device=batch[inputs_key].device
        )
        modelling_mask = modelling_mask.ge(1-self.masking_rate)
        modelling_mask = torch.where(
            batch['attention_mask'] == 1,
            modelling_mask,
            torch.zeros_like(modelling_mask)
        )
        batch['modelling_mask'] = modelling_mask.unsqueeze(2).repeat(
            1,
            1,
            self.in_dim
        ).to(torch.long)
        batch['masked_inputs'] = torch.masked_select(
            input=batch[inputs_key],
            mask=batch['modelling_mask'].to(torch.bool)
        ).detach().clone()
        batch['inputs_embeds'] = torch.where(
            batch['modelling_mask'] == 1,
            self.msk_embed.repeat(
                batch[inputs_key].size()[0],
                batch[inputs_key].size()[1],
                1
            ),
            batch[inputs_key].to(torch.float)
        )
        
        return batch