
#/usr/bin/env python3

from typing import Dict
import torch
from src.embedder.msm import MSMEmbedder


class BERTEmbedder(MSMEmbedder):
    
    def __init__(
        self,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = 'BERTEmbedder'
        self.training_style = 'BERT'
        assert self.training_style in {'BERT', 'decoding'}, f'{self.training_style} not supported'
        self._root_training_style = 'BERT'
        self.sep_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, self.in_dim),
                requires_grad=True
            )
        )
        self._embeds.append(self.sep_embed)
        self._init_embeds()

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        batch_out = dict(batch)
        labels =  torch.clone(batch['labels']) if 'labels' in batch else None
        
        if self.training_style == 'BERT':
            batch_out = self.mask_inputs(batch=batch_out)
            batch_out = self.add_sep_embeds(batch=batch_out)
        
        batch_out = self.add_cls_embed(batch=batch_out)
        
        if labels is not None:
            batch_out['labels'] = labels
        
        return batch_out

    def add_sep_embeds(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        """add sep token embed at the end of each 
        of two sequences contained in inputs""" 
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        assert inputs_key in batch, f'{inputs_key} not found in batch'
        assert 'token_type_ids' in batch, 'token_type_ids not found in batch'
        inputs_shape = batch[inputs_key].size()
        batch_size = inputs_shape[0]
        device = batch[inputs_key].device
        inputs_embeds = []
        t_rs = []
        token_type_ids = []
        if 'modelling_mask' in batch:
            modelling_mask = []
        
        for i in range(batch_size):
            len_seq_a = sum(
                torch.masked_select(
                    batch['token_type_ids'][i],
                    batch['attention_mask'][i].to(torch.bool)
                ) == 0
            ) 
            len_seq_b = sum(
                torch.masked_select(
                    batch['token_type_ids'][i],
                    batch['attention_mask'][i].to(torch.bool)
                ) == 1
            ) 
            inputs_embeds.append(
                torch.cat(
                    (
                        batch[inputs_key][i,:len_seq_a],
                        self.sep_embed,
                        batch[inputs_key][i,len_seq_a:(len_seq_a+len_seq_b)],
                        self.sep_embed,
                        # and pad with zeros
                        torch.zeros(
                            size=(
                                inputs_shape[1]-len_seq_a-len_seq_b,
                                inputs_shape[-1]
                            ),
                            device=device
                        )
                    ),
                    dim=0
                )
            )
            t_rs.append(
                torch.cat(
                    (
                        batch["t_rs"][i,:len_seq_a],
                        torch.ones(1, device=device) * -1, # idx for dummy t_r embedding
                        batch["t_rs"][i,len_seq_a:(len_seq_a+len_seq_b)],
                        torch.ones(1, device=device) * -1,
                        # and pad with zeros
                        torch.zeros(
                            inputs_shape[1]-len_seq_a-len_seq_b,
                            device=device
                        )
                    ),
                    dim=0
                )
            )
            token_type_ids.append(
                torch.cat(
                    (
                        batch["token_type_ids"][i,:len_seq_a],
                        torch.zeros(1, device=device),
                        batch["token_type_ids"][i,len_seq_a:(len_seq_a+len_seq_b)],
                        torch.ones(1, device=device),
                        # and pad with zeros
                        torch.zeros(
                            inputs_shape[1]-len_seq_a-len_seq_b,
                            device=device
                        )
                    ),
                    dim=0
                )
            )

            if 'modelling_mask' in batch:
                modelling_mask.append(
                    torch.cat(
                        (
                            batch["modelling_mask"][i,:len_seq_a],
                            # we dont try to reconstruct sep embeds
                            torch.zeros(
                                size=(
                                    1,
                                    self.in_dim
                                ),
                                device=device
                            ),
                            batch["modelling_mask"][i,len_seq_a:(len_seq_a+len_seq_b)],
                            torch.zeros(
                                size=(
                                    1,
                                    self.in_dim
                                ),
                                device=device
                            ),
                            # and pad with zeros
                            torch.zeros(
                                size=(
                                    inputs_shape[1]-len_seq_a-len_seq_b,
                                    self.in_dim
                                ),
                                device=device
                            )
                        ),
                        dim=0
                    )
                )
        
        batch['inputs_embeds'] = torch.stack(
            inputs_embeds,
            dim=0
        )
        batch['t_rs'] = torch.stack(
            t_rs,
            dim=0
        )
        batch['token_type_ids'] = torch.stack(
            token_type_ids,
            dim=0
        )
        
        if 'modelling_mask' in batch:
            batch['modelling_mask'] = torch.stack(
                modelling_mask,
                dim=0
            )

        batch['attention_mask'] = self._pad_tensor_left_by_n(
            tensor=batch['attention_mask'],
            n=2,
            pad_value=1
        )
        
        return batch

    def is_next_loss(
        self,
        is_next_logits,
        is_next
        ) -> Dict[str, torch.tensor]:
        
        return {
            'is_next_loss': self.bxe_loss(
                input=is_next_logits.ravel(),
                target=is_next.to(torch.float).ravel()
            )
        }

    def _root_loss(
        self,
        masked_inputs,
        is_next,
        outputs,
        is_next_logits,
        modelling_mask,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        loss = self.masking_loss(
            masked_inputs=masked_inputs,
            outputs=outputs,
            modelling_mask=modelling_mask
        )
        loss.update(
            self.is_next_loss(
                is_next_logits=is_next_logits,
                is_next=is_next
            )
        )
        
        return loss