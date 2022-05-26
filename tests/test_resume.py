#!/usr/bin/env python3

import os
import torch
import shutil
from tests import get_all_architectures_with_all_train_styles,\
    run_tests,\
    run_train_process,\
    DEFAULT_CONFIG


N_STEPS = 1
LAST_CHECKPOINT = f"checkpoint-{N_STEPS}"
RUN_NAME = 'test_resume'
LOG_DIR = f'tests/.cache/results/{RUN_NAME}'
TRAIN_CONFIG = dict(DEFAULT_CONFIG)
TRAIN_CONFIG = {
    **DEFAULT_CONFIG,
    "data": "tests/data/tarfiles",
    "training_steps": N_STEPS,
    "validation_steps": 2,
    "log_every_n_steps": 1,
    "per_device_training_batch_size": 2,
    "per_device_validation_batch_size": 3,
    'num_hidden_layers': 3,
    'embedding_dim': 768, 
    "wandb_mode": "disabled",
    "decoding_target": "task_label", # MDTB test data
    "num_decoding_classes": 26, # MDTB test data
    "fp16": False,
    "log_dir": LOG_DIR,
    'run_name': RUN_NAME
}


def test_resume() -> None:

    if os.path.exists(LOG_DIR):
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)
    
    for architecture, training_style in get_all_architectures_with_all_train_styles():
        print(
            f'Testing resume for {architecture}-architecture '
            f'and {training_style}-training'
        )
        trainer = run_train_process(
            config={
                **TRAIN_CONFIG,
                'architecture': architecture,
                'training_style': training_style
            }
        )
        model = trainer.model
        trainer_after_restart = run_train_process(
            config={
                **TRAIN_CONFIG,
                'resume_from': LOG_DIR
            }
        )
        loaded_model = trainer_after_restart.model
        assert model.state_dict().keys() == loaded_model.state_dict().keys()
        for key in model.state_dict().keys():
            assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key])
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)

if __name__ == "__main__":
    run_tests()