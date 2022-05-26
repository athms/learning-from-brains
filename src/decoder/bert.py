#!/usr/bin/env python3

from typing import Dict
import torch
from transformers import BertConfig, BertModel
from src.decoder.gpt import GPTModel


class BERTModel(GPTModel):
    def __init__(
        self,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = 'BERT'
        self.config = BertConfig(
            vocab_size=1,
            max_position_embeddings=self.n_positions,
            hidden_size=self.embed_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.embed_dim * self.intermediate_dim_factor,
            hidden_act=self.hidden_activation,
            attention_probs_dropout_prob = self.dropout_attn,
            hidden_dropout_prob = self.dropout_resid,
        )
        self.transformer = BertModel(
            config=self.config,
            add_pooling_layer=False
        )
        self.is_next_head = torch.nn.Sequential(
            torch.nn.Linear(
                self.embed_dim,
                1
            )
        )
        #self.add_pooler_layer()

    def is_next_pred(
        self,
        outputs: torch.tensor
        ) -> torch.tensor:
        is_next_outputs = {'pooler_outputs': self.pooler_layer(outputs[:,0,:])}
        is_next_outputs['is_next_logits'] =  self.is_next_head(is_next_outputs['pooler_outputs'])
        return is_next_outputs

    def decode(
        self,
        outputs: torch.tensor
        ) -> Dict[str, torch.tensor]:
        assert self.is_decoding_mode, 'GPTModel must be in decoding_mode.'
        assert self.pooler_layer is not None, 'pooler_layer head must be added.'
        assert self.decoding_head is not None, 'decoding head must be added.'
        decoding_outputs = {'pooler_outputs': self.pooler_layer(outputs[:,0,:])}
        decoding_outputs['decoding_logits'] = self.decoding_head(decoding_outputs['pooler_outputs'])
        return decoding_outputs

    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]: 
        transformer_outputs = self.transformer.forward(
            inputs_embeds=batch['inputs_embeds'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', None),
            return_dict=True
        )
        outputs = {'outputs': transformer_outputs['last_hidden_state']}

        if not self.is_decoding_mode:
            outputs.update(self.is_next_pred(outputs=outputs['outputs']))
        
        else:
            outputs.update(self.decode(outputs=outputs['outputs']))
        
        return outputs


class PretrainedBERT(BERTModel):
    def __init__(
        self,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = 'PretrainedBERT'
        self.config = BertConfig()
        self.n_positions = self.config.max_position_embeddings
        self.embed_dim = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads
        self.intermediate_dim_factor = 4
        self.hidden_activation = self.config.hidden_act
        self.dropout_attn = self.config.attention_probs_dropout_prob
        self.dropout_resid = self.config.hidden_dropout_prob
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.is_next_head = torch.nn.Sequential(
            torch.nn.Linear(
                self.embed_dim,
                1
            )
        )
        self.add_pooler_layer()