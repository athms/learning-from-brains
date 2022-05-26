#!/usr/bin/env python3

def make_batcher(
    training_style: str='CSM',
    seq_min: int=10,
    seq_max: int=50,
    bert_seq_gap_min: int=1,
    bert_seq_gap_max: int=5,
    decoding_target: str=None,
    sample_random_seq: bool=True,
    seed: int=None,
    bold_dummy_mode: bool=False,
    ):
    
    kwargs = {
        "seq_min": seq_min,
        "seq_max": seq_max,
        "gap_min": bert_seq_gap_min,
        "gap_max": bert_seq_gap_max,
        "decoding_target": decoding_target,
        "sample_random_seq": sample_random_seq,
        "seed": seed,
        "bold_dummy_mode": bold_dummy_mode
    }
    sample_keys = [
        'inputs',
        'attention_mask',
        't_rs'
    ]
    if training_style in {'BERT', 'NetBERT'}:
        sample_keys.extend(
            [
                'token_type_ids',
                'is_next'
            ]
        )
        from src.batcher.bert import BERTBatcher
        return BERTBatcher(**{**kwargs, **{'sample_keys': sample_keys}})

    elif training_style in {'CSM', 'MSM', 'MNM', 'autoencoder'}:
        from src.batcher.base import BaseBatcher
        return BaseBatcher(**{**kwargs, **{'sample_keys': sample_keys}})

    elif training_style == 'decoding':
        sample_keys.append('labels')
        from src.batcher.base import BaseBatcher
        return BaseBatcher(**{**kwargs, **{'sample_keys': sample_keys}})

    else:
        raise ValueError('unknown training style.')