#!/usr/bin/env python3

import os
import shutil
from tests import get_architectures_with_pretrain_styles
from tests import run_tests, run_train_process, DEFAULT_CONFIG


RUN_NAME = 'test_embed_dims'
LOG_DIR = f'tests/.cache/results/{RUN_NAME}'
EMBED_DIMS = [192, 384, 768, 1024]
TRAIN_CONFIG = {
    **DEFAULT_CONFIG,
    "data": "tests/data/tarfiles",
    "training_steps": 2,
    "validation_steps": 2,
    "log_every_n_steps": 1,
    'num_hidden_layers': 2,
    'num_attention_heads': 2,
    'decoding_target': "task_label",
    'num_decoding_classes': 26,
    "per_device_training_batch_size": 2,
    "per_device_validation_batch_size": 2,
    "wandb_mode": "disabled",
    "fp16": False,
    "log_dir": LOG_DIR,
    'run_name': RUN_NAME
}


def test_embed_dims():
    
    if os.path.exists(LOG_DIR):
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)

    for embed_dim in EMBED_DIMS:

        for architecture, training_style in get_architectures_with_pretrain_styles():
            print(
                f'Testing {embed_dim}-dim embedding for '
                f'{architecture}-architecture and '
                f'{training_style}-training'
            )
            test_run_config = {
                **TRAIN_CONFIG,
                'embedding_dim': embed_dim, 
                'architecture': architecture,
                'training_style': training_style
            }
            trainer = run_train_process(config=test_run_config)
            model = trainer.model
            train_dataloader = trainer.get_train_dataloader()
            batch = next(iter(train_dataloader))
            batch = trainer._move_batch_to_device(batch=batch)
            inputs_dim = batch['inputs'].size()[-1]
            batch_prepared = model.embedder.prep_batch(batch=batch)
            batch_prepared["inputs_embeds"] = model.embedder(batch=batch_prepared)
            inputs_embeds_dim = batch_prepared['inputs_embeds'].size()[-1]
            assert inputs_embeds_dim == embed_dim, (
                'inputs_embeds have wrong dim; '
                f'is {inputs_embeds_dim} should be {embed_dim}'
            )
            outputs_decoder = model.decoder(batch=batch_prepared)
            outputs_decoder_dim = outputs_decoder['outputs'].size()[-1]
            assert outputs_decoder_dim == embed_dim, (
                'decoder outputs size does not match embedding dim; '
                f'is {outputs_decoder_dim} should be {embed_dim}'
            )
            outputs = model(batch=batch)
            outputs_dim = outputs['outputs'].size()[-1]
            assert outputs_dim == inputs_dim, (
                'model outputs size does not match inputs size; '
                f'is {outputs_dim} should be {inputs_dim}'
            )

            print(
                f"Removing {LOG_DIR}..."
            )
            shutil.rmtree(LOG_DIR)
    

if __name__ == "__main__":
    run_tests()