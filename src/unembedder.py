#!/usr/bin/env python3

import torch
from einops import rearrange


class UnEmbedder(torch.nn.Module):
    """
    Unmebedding model; used to project predicted 
    output sequences of src.decoder back to input 
    space during upstream learning.
    
    Args
    ----
    embed_dim: int
        Dimension of the embedding space.
    out_dim: int
        Dimension of the output space.
    num_hidden_layers: int
        Number of hidden layers of projection model.
        If >1, all hidden layers except for the last
        are activated with Gelu activation.
    dropout: float
        Dropout ratio for the projection model.

    Core methods
    ----
    forward(inputs, **kwargs)
        Projection of input to output space.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        out_dim: int = 1024,
        num_hidden_layers: int = 1,
        dropout: int = 0.1,
        ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        layer_stack = []
        for _ in range(self.num_hidden_layers-1):
            layer_stack.extend(
                [
                    torch.nn.Linear(
                        in_features=self.embed_dim,
                        out_features=self.embed_dim
                    ),
                    torch.nn.LayerNorm(self.embed_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(p=self.dropout)
                ]
            )
        layer_stack.extend(
            [
                torch.nn.Linear(
                    in_features=self.embed_dim,
                    out_features=self.out_dim
                )
            ]
        )
        self.model = torch.nn.Sequential(*layer_stack)

    def stack_inputs(
        self,
        tensor
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='b s e -> (b s) e'
        )

    def unstack_inputs(
        self,
        tensor,
        b
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='(b s) e -> b s e',
            b=b
        )

    def forward(
        self,
        inputs,
        **kwargs
        ) -> torch.tensor:
        inputs_stacked = self.stack_inputs(tensor=inputs)
        
        return {
            'outputs': self.unstack_inputs(
                tensor=self.model(inputs_stacked),
                b=inputs.size()[0]
            )
        }


def make_unembedder(
    embed_dim: int = 768,
    out_dim: int = 1024,
    num_hidden_layers: int = 1,
    dropout: int = 0.1
    ) -> torch.nn.Module:
    """
    Creates a UnEmbedder object.

    Args
    ----
    embed_dim: int
        Dimension of the embedding space.
    out_dim: int
        Dimension of the output space.
    num_hidden_layers: int
        Number of hidden layers of projection model.
        If >1, all hidden layers except for the last
        are activated with Gelu activation.
    dropout: float
        Dropout ratio for the projection model.

    Core methods
    ----
    forward(inputs, **kwargs)
        Projection of input to output space.
    """
    return UnEmbedder(
        embed_dim=embed_dim,
        out_dim=out_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout
    )