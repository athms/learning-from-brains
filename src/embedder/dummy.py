#/usr/bin/env python3

import torch
from typing import Dict


class DummyEmbedder(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.name = 'DummyEmbedder'
        self.training_style = 'dummy'
        self._root_training_style = 'dummy'
        self.is_decoding_mode = False
        self.xe_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.bxe_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def switch_decoding_mode(self, is_decoding_mode: bool=False) -> None:
        self.is_decoding_mode = is_decoding_mode
        return None

    def embed_inputs(
        self,
        inputs: torch.tensor
        ) -> torch.tensor:
        return inputs
    
    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> torch.tensor:
        return batch['inputs']

    def decoding_loss(
        self,
        decoding_logits,
        labels,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        
        return {
            'decoding_loss': self.xe_loss(
                input=decoding_logits,
                target=labels.to(dtype=torch.long)
            )
        }
    
    def reconstruction_loss(
        self,
        input,
        target,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        
        return {
            'reconstruction_loss': self.l1_loss(
                input=input,
                target=target
            )
        }

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict:
        batch_out = {}
        
        for key in batch:
            
            if (
                torch.is_tensor(batch[key])
                and key != 'labels'
            ):
                batch_out[key] = batch[key].to(torch.float)
            
            elif key == 'labels':
                batch_out[key] = batch['labels'].to(torch.int)

            else:
                batch_out[key] = torch.clone(batch[key])
        
        # dummy copy of inputs to be used in forward pass
        batch_out['inputs_embeds'] = torch.clone(batch_out['inputs'])
        
        return batch_out

    def _root_loss(
        self,
        inputs,
        outputs,
        attention_mask,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        attention_mask = torch.unsqueeze(attention_mask, -1).repeat(1,1,self.in_dim)
        
        return  self.reconstruction_loss(
            input=torch.masked_select(outputs, attention_mask.to(torch.bool)),
            target=torch.masked_select(inputs, attention_mask.to(torch.bool))
        )

    def loss(
        self,
        batch,
        outputs
        ) -> Dict[str, torch.tensor]:

        if self.is_decoding_mode:
            losses = self.decoding_loss(
                **batch,
                **outputs
            )
        
        else:
            losses = self._root_loss(
                **batch,
                **outputs
            )

        if 'loss' not in losses:
            losses['loss'] = sum(losses.values())

        return losses