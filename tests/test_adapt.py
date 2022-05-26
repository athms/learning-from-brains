#!/usr/bin/env python3

import os
import shutil
import torch
from tests import get_architectures_with_pretrain_styles,\
    run_tests,\
    run_train_process,\
    DEFAULT_CONFIG


N_STEPS = 10
LAST_CHECKPOINT = f"checkpoint-{N_STEPS}"
RUN_NAME = 'test_from-pretrained'
LOG_DIR = f'tests/.cache/results/{RUN_NAME}'
PRETRAIN_CONFIG = {
    **DEFAULT_CONFIG,
    "data": "tests/data/tarfiles",
    "training_steps": N_STEPS,
    "validation_steps": 2,
    "log_every_n_steps": 1,
    "per_device_training_batch_size": 2,
    "per_device_validation_batch_size": 3,
    'num_hidden_layers': 2,
    'embedding_dim': 768,
    "wandb_mode": "disabled",
    "fp16": False,
    "log_dir": LOG_DIR,
    'run_name': RUN_NAME,
}
ADAPT_CONFIG = {
    **PRETRAIN_CONFIG,
    "training_style": "decoding",
    "decoding_target": "task_label",
    "num_decoding_classes": 26,
    "sample_random_seq": False,
    "seq_max": 50
}


def test_adapt_decoding() -> None:

    if os.path.exists(LOG_DIR):
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)

    for architecture, pre_training_style in get_architectures_with_pretrain_styles():
        print(
            f'Testing adapt for {architecture}-architecture '
            f'and {pre_training_style}-pre-training'
        )

        # pretrain model
        pretrainer = run_train_process(
            config={
                    **PRETRAIN_CONFIG,
                    'architecture': architecture,
                    'training_style': pre_training_style,
                }
            )
        # adapt mdoel
        adapt_trainer = run_train_process(
            config={
                    **ADAPT_CONFIG,
                    'architecture': architecture,
                    'pretrained_model': f"{LOG_DIR}/model_final/pytorch_model.bin",
                    'do_train': False
                }
            )
            
    print(
        f"Removing {LOG_DIR}..."
    )
    shutil.rmtree(LOG_DIR)

if __name__ == "__main__":
    run_tests()