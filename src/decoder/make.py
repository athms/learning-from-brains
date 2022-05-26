#!/usr/bin/env python3


def make_decoder(
    architecture: str='GPT',
    num_hidden_layers: int = 4,
    embed_dim: int = 768,
    output_dim: int = 1024,
    num_attention_heads: int = 12,
    intermediate_dim_factor: int=4,
    n_positions: int = 512,
    hidden_activation: str='gelu_new',
    dropout: float = 0.1,
    autoen_teacher_forcing_ratio: float = 0.5
    ):

    kwargs = {
        "num_hidden_layers": num_hidden_layers,
        "embed_dim": embed_dim,
        "output_dim": output_dim,
        "num_attention_heads": num_attention_heads,
        "intermediate_dim_factor": intermediate_dim_factor,
        "n_positions": n_positions,
        "hidden_activation": hidden_activation,
        "dropout": dropout,
        "teacher_forcing_ratio": autoen_teacher_forcing_ratio
    }

    if architecture == 'autoencoder':
        from src.decoder.autoencoder import AutoEncoder
        return AutoEncoder(**kwargs)
    
    elif architecture in {'BERT', 'NetBERT'}:
        from src.decoder.bert import BERTModel
        return BERTModel(**kwargs)
    
    elif architecture == 'GPT':
        from src.decoder.gpt import GPTModel
        return GPTModel(**kwargs)

    elif architecture == 'PretrainedGPT2':
        from src.decoder.gpt import PretrainedGPT2
        return PretrainedGPT2(**kwargs)

    elif architecture == 'PretrainedBERT':
        from src.decoder.bert import PretrainedBERT
        return PretrainedBERT(**kwargs)

    elif architecture == 'LogisticRegression':
        from src.decoder.logistic_regression import LogisticRegression
        return LogisticRegression(**kwargs)
    
    else:
        raise ValueError(f'{architecture}-architecture unkown.')