#!/usr/bin/env python3 

import torch
from typing import Dict
import warnings


class Model(torch.nn.Module):
    def __init__(
        self,
        embedder: torch.nn.Module,
        decoder: torch.nn.Module,
        unembedder: torch.nn.Module = None
        ) -> torch.nn.Module:
        super().__init__()
        self.name = f'Embedder-{embedder.name}_Decoder-{decoder.name}'
        self.embedder = embedder
        self.decoder = decoder
        self.unembedder = unembedder
        self.is_decoding_mode = False

    def from_pretrained(
        self,
        pretrained_path
        ) -> None:
        print(
            f'Loading pretrained model from {pretrained_path}'
        )

        if next(self.parameters()).is_cuda:
            pretrained = torch.load(pretrained_path)

        else:
            pretrained = torch.load(pretrained_path, map_location=torch.device('cpu'))
        
        for k in self.state_dict():
            
            if k in pretrained:
                assert pretrained[k].shape == self.state_dict()[k].shape,\
                    f'{k} shape mismatch between pretrained model and current model '+\
                    f'{pretrained[k].shape} vs {self.state_dict()[k].shape}'
        
        for k in pretrained:
            
            if k not in self.state_dict():
                warnings.warn(
                    f'Warning: /!\ Skipping {k} from {pretrained_path} '\
                    'because it is not part of the current model'
                )

        # we set strcit=False, because we can be sure
        # that all relevant keys are in pretrained
        self.load_state_dict(pretrained, strict=False)

    def switch_decoding_mode(
        self,
        is_decoding_mode: bool = False,
        num_decoding_classes: int = None
        ) -> None:
        self.is_decoding_mode = is_decoding_mode
        self.embedder.switch_decoding_mode(is_decoding_mode=is_decoding_mode)
        self.decoder.switch_decoding_mode(
            is_decoding_mode=is_decoding_mode,
            num_decoding_classes=num_decoding_classes
        )

    def compute_loss(
        self,
        batch,
        return_outputs: bool = False
        ) -> Dict[str, torch.tensor]:
        (outputs, batch) = self.forward(
            batch=batch,
            return_batch=True
        )
        losses = self.embedder.loss(
            batch=batch,
            outputs=outputs
        )

        return (losses, outputs) if return_outputs else losses

    def prep_batch(
        self,
        batch
        ) -> Dict[str, torch.tensor]:
        return self.embedder.prep_batch(batch=dict(batch))

    def forward(
        self,
        batch: Dict[str, torch.tensor],
        prep_batch: bool = True,
        return_batch: bool = False
        ) -> torch.tensor:
        
        if prep_batch:
            batch = self.prep_batch(batch=batch)
        
        else:
            assert 'inputs_embeds' in batch, 'inputs_embeds not in batch'

        batch['inputs_embeds'] = self.embedder(batch=batch)
        outputs = self.decoder(batch=batch)
        
        if self.unembedder is not None and not self.is_decoding_mode:
            outputs['outputs'] = self.unembedder(inputs=outputs['outputs'])['outputs']
        
        return (outputs, batch) if return_batch else outputs