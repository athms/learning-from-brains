#!/usr/bin/env python3

import os
import shutil
from tests import get_architectures_with_pretrain_styles,\
    get_all_architectures_with_decoding,\
    run_tests,\
    run_train_process,\
    DEFAULT_CONFIG


RUN_NAME = 'test_train'
LOG_DIR = f'tests/.cache/results/{RUN_NAME}'
TRAIN_CONFIG = {
    **DEFAULT_CONFIG,
    "data": "tests/data/tarfiles",
    "training_steps": 2,
    "validation_steps": 2,
    "log_every_n_steps": 1,
    'num_hidden_layers': 2,
    'num_attention_heads': 2,
    "per_device_training_batch_size": 2,
    "per_device_validation_batch_size": 2,
    "wandb_mode": "disabled",
    "fp16": False,
    "log_dir": LOG_DIR,
    'run_name': RUN_NAME
}


def test_pretrain():
    
    if os.path.exists(LOG_DIR):
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)
    
    for architecture, training_style in get_architectures_with_pretrain_styles():
        print(
            f'Testing train for {architecture}-architecture '
            f'and {training_style}-training'
        )
        test_run_config = {
            **TRAIN_CONFIG,
            'architecture': architecture,
            'training_style': training_style
        }
        _ = run_train_process(config=test_run_config)
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)


def test_decoding():
    training_style = 'decoding'
        
    if os.path.exists(LOG_DIR):
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)
    
    for architecture, _ in get_all_architectures_with_decoding():
        print(
            f'Testing train for {architecture}-architecture '
            f'and {training_style}-training'
        )
        test_run_config = {
            **TRAIN_CONFIG,
            'training_style': training_style,
            "decoding_target": "task_label", # MDTB test data
            "num_decoding_classes": 26, # MDTB test data
            'architecture': architecture
        }
    
        _ = run_train_process(config=test_run_config)
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)


if __name__ == "__main__":
    run_tests()