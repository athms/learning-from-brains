#/usr/bin/env python3

from typing import Dict
import torch
import random
import warnings


class Encoder(torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 4,
        embed_dim: int = 768,
        dropout: float = 0.2,
        **kwargs
        ) -> None:
        super(Encoder, self).__init__()
        self.name = 'Encoder'
        self.num_hidden_layers = num_hidden_layers
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.pooler_layer = torch.nn.Sequential(
            # also applies dropout to last outputs of lstms!
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(
                in_features=self.embed_dim,
                out_features=self.embed_dim
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(self.dropout),
        )
        self.lstms = torch.nn.ModuleList(
            [
                torch.nn.LSTM(
                    input_size=self.embed_dim,
                    hidden_size=self.embed_dim,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0. # no dropout here as we apply dropout to the normalization layer
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        self.layer_norms = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(self.embed_dim),
                    torch.nn.Dropout(p=self.dropout)
                )
                for _ in range(self.num_hidden_layers-1)
            ]
        )

    def forward(
        self,
        inputs: torch.tensor = None,
        hidden: torch.tensor = None,
        cell_state: torch.tensor = None,
        **kwargs
        ) -> torch.Tensor:
        
        if hidden is None:
            hidden = torch.zeros(
                (
                    2, # as bidirectional
                    inputs.size()[0],
                    self.embed_dim
                ),
                device=inputs.device
            )
        
        if cell_state is None:
            cell_state = torch.zeros(
                (
                    2, # as bidirectional
                    inputs.size()[0],
                    self.embed_dim
                ),
                device=inputs.device
            )

        for i, lstm in enumerate(self.lstms):
            
            # skip connection inspired by:
            # https://arxiv.org/pdf/1609.08144.pdf
            if i > 0:
                inputs = self.layer_norms[i-1](inputs + prev_inputs)
            
            lstm.flatten_parameters()
            outputs, (hidden, cell_state) = lstm(inputs, (hidden, cell_state))
            # average outputs over bidirectional modules
            outputs = (
                outputs[:,:,:int(outputs.size()[-1]/2)] +
                outputs[:,:,int(outputs.size()[-1]/2):]
            ) / 2.
            prev_inputs, inputs = inputs, outputs
        
        hidden = torch.mean(hidden, dim=0)
        return {
            'outputs': outputs,
            # final encoding is defined by the first sequence element
            # (as input vectors padded with zeros to the right)
            'pooler_outputs': self.pooler_layer(hidden), # outputs[:,0]
            # average hidden state over bidirectional modules
            'hidden': hidden.unsqueeze(0),
        }


class Decoder(torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 4,
        embed_dim: int = 768,
        dropout: float = 0.2,
        teacher_forcing_ratio: float = 0.5,
        **kwargs
        ) -> None:
        super(Decoder, self).__init__()
        self.name = 'Decoder'
        self.num_hidden_layers = num_hidden_layers
        self.embed_dim = embed_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = dropout
        # self.sos_embed = torch.nn.Parameter(
        #     torch.empty(
        #         size=(1, 1, self.embed_dim)
        #     )
        # )
        # torch.nn.init.normal_(
        #     tensor=self.sos_embed,
        #     mean=0.0,
        #     std=1.0,
        # )
        self.lstms = torch.nn.ModuleList(
            [
                torch.nn.LSTM(
                    input_size=self.embed_dim,
                    hidden_size=self.embed_dim,
                    num_layers=1,
                    bidirectional=False, # decoder is not bidirectional!
                    batch_first=True,
                    dropout=0. # no dropout here as we apply dropout to the normalization layer
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        self.layer_norms = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(self.embed_dim),
                    torch.nn.Dropout(p=self.dropout)
                )
                for _ in range(self.num_hidden_layers-1)
            ]
        )
        self.linear_output_layer = torch.nn.Sequential(
            # we also apply dropout to last output of lstms!
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(
                in_features=self.embed_dim,
                out_features=self.embed_dim
            ),
            torch.nn.Dropout(self.dropout)
        )

    def forward_step(
        self,
        inputs: torch.tensor = None,
        hidden: torch.tensor = None,
        cell_state: torch.tensor = None,
        **kwargs
        ) -> torch.Tensor:
        
        if hidden is None:
            hidden = torch.zeros(
                (
                    1, # as not bidirectional
                    inputs.size()[0],
                    self.embed_dim
                ),
                device=inputs.device
            )
        
        if cell_state is None:
            cell_state = torch.zeros(
                (
                    1, # as not bidirectional
                    inputs.size()[0],
                    self.embed_dim
                ),
                device=inputs.device
            )

        for i, lstm in enumerate(self.lstms):
            
            # skip connection inspired by:
            # https://arxiv.org/pdf/1609.08144.pdf
            if i > 0:
                inputs = self.layer_norms[i-1](inputs + prev_inputs)
            
            lstm.flatten_parameters()
            outputs, (hidden, cell_state) = lstm(inputs, (hidden, cell_state))
            prev_inputs, inputs = inputs, outputs
        
        # we pass entire output through output layer,
        # as decoder tries to reconstruct full input
        outputs = self.linear_output_layer(outputs)
        
        return outputs, hidden, cell_state

    def forward(
        self,
        inputs: torch.tensor,
        targets: torch.tensor,
        hidden: torch.tensor = None,
        cell_state: torch.tensor = None,
        #encoder_outputs: torch.tensor = None,
        ) -> torch.Tensor:
        
        if self.training:
            use_teacher_forcing = random.random() < self.teacher_forcing_ratio
            
        else:
            use_teacher_forcing = False
        
        # if not use_teacher_forcing:
        #     inputs = self.sos_embed.repeat(
        #         inputs.size()[0],
        #         1,
        #         1
        #     )

        outputs = []

        for i in range(targets.size()[1]):
            
            if use_teacher_forcing:
                # teacher forcing: feed target as next input
                out, hidden, cell_state = self.forward_step(
                    inputs=targets[:,i].unsqueeze(1),
                    hidden=hidden,
                    cell_state=cell_state,
                )
                outputs.append(out)

            else:
                # no teacher forcing: use own predictions as next input
                inputs, hidden, cell_state = self.forward_step(
                    inputs=inputs, 
                    hidden=hidden, 
                    cell_state=cell_state,
                )
                outputs.append(inputs)
        
        return {'outputs': torch.cat(outputs, dim=1)}


class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 4,
        embed_dim: int = 768,
        output_dim: int = 1024,
        dropout: float = 0.2,
        teacher_forcing_ratio: float = 0.5,
        **kwargs
        ) -> None:
        super(AutoEncoder, self).__init__()
        self.name = 'Autoencoder'
        self.num_hidden_layers = num_hidden_layers
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = Encoder(
            num_hidden_layers=self.num_hidden_layers,
            embed_dim=self.embed_dim,
            dropout=self.dropout
        )
        self.decoder = Decoder(
            num_hidden_layers=self.num_hidden_layers,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
            teacher_forcing_ratio=self.teacher_forcing_ratio
        )        
        self.is_decoding_mode = False
        self.decoding_head = None
        self.num_decoding_classes = None

    def switch_decoding_mode(
        self,
        is_decoding_mode: bool=False,
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
            warnings.warn(
                f'Warning: overwriting existing {num_decoding_classes}-class decoding head.'
            )
        
        self.num_decoding_classes = num_decoding_classes
        self.decoding_head = torch.nn.Sequential(
            # no need for dropout here as we apply dropout to pooler_layer
            torch.nn.Linear(
                in_features=self.embed_dim,
                out_features=self.num_decoding_classes
            )
        )
    
    def decode(
        self,
        pooler_outputs: torch.tensor = None,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        assert self.is_decoding_mode, 'autoencoder must be in decoding_mode.'
        assert self.decoding_head is not None, 'decoding head must be added.'
        return {'decoding_logits': self.decoding_head(pooler_outputs)}
        
    def forward(
        self,
        batch: Dict[str, torch.tensor],
        ) -> torch.Tensor:
        outputs_encoder = self.encoder(inputs=batch['inputs_embeds'])
        outputs = {'pooler_outputs': outputs_encoder['pooler_outputs']}

        if not self.is_decoding_mode:
            outputs_decoder = self.decoder(
                inputs=outputs_encoder['pooler_outputs'].unsqueeze(1),
                hidden=outputs_encoder['hidden'],
                targets=batch['inputs_embeds'],
            )
            outputs['outputs'] = outputs_decoder['outputs']

        else:
            outputs.update(
                {
                    **self.decode(pooler_outputs=outputs['pooler_outputs'])
                }
            )
        
        return outputs