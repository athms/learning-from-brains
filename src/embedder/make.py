#!/usr/bin/env python3


def make_embedder(
    architecture: str='GPT',
    training_style: str='CSM',
    in_dim: int=1024,
    embed_dim: int=768,
    num_hidden_layers: int=1,
    masking_rate: float=0.15,
    dropout: float=0.1,
    t_r_precision: int = 0.2, # in seconds
    max_t_r: int = 300, # in seconds (= 10min)
    n_positions: int=512
    ):

    kwargs = {
        "in_dim": in_dim,
        "embed_dim": embed_dim,
        "num_hidden_layers": num_hidden_layers,
        "dropout": dropout,
        "t_r_precision": t_r_precision,
        "max_t_r": max_t_r,
        "n_positions": n_positions
    }

    if training_style == 'autoencoder':
        from src.embedder.autoen import AutoenEmbedder
        embedder = AutoenEmbedder(**kwargs)

    elif training_style == 'CSM':
        from src.embedder.csm import CSMEmbedder
        embedder = CSMEmbedder(**kwargs)

    elif training_style == 'MSM':
        from src.embedder.msm import MSMEmbedder
        embedder = MSMEmbedder(**kwargs)

    elif training_style == 'MNM':
        from src.embedder.mnm import MNMEmbedder
        embedder = MNMEmbedder(**kwargs)
    
    elif training_style == 'BERT':
        from src.embedder.bert import BERTEmbedder
        embedder = BERTEmbedder(**kwargs)
    
    elif training_style == 'NetBERT':
        from src.embedder.netbert import NetBERTEmbedder
        embedder = NetBERTEmbedder(**kwargs)

    elif training_style == 'decoding':

        if architecture == 'autoencoder':
            from src.embedder.autoen import AutoenEmbedder
            embedder = AutoenEmbedder(**kwargs)

        elif architecture in {'GPT', 'PretrainedGPT2'}:
            from src.embedder.csm import CSMEmbedder
            embedder = CSMEmbedder(**kwargs)
        
        elif architecture in {
            'BERT',
            'PretrainedBERT',
            'PretrainedDistilBERT',
            'PretrainedRoBERTa'
        }:
            from src.embedder.bert import BERTEmbedder
            embedder = BERTEmbedder(**kwargs)

        elif architecture == 'NetBERT':
            from src.embedder.netbert import NetBERTEmbedder
            embedder = NetBERTEmbedder(**kwargs)

        elif architecture == 'LogisticRegression':
            from src.embedder.dummy import DummyEmbedder
            embedder = DummyEmbedder(**kwargs)

        else:
            raise ValueError('unkown architecture')

    else:
        raise ValueError('unknown training style.')
    
    return embedder