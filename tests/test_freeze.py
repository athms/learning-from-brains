#!/usr/bin/env python3

import os
import shutil
import numpy as np
import torch
from tests import get_all_architectures_with_all_train_styles,\
    run_tests,\
    run_train_process,\
    DEFAULT_CONFIG


N_STEPS = 3
LAST_CHECKPOINT = f"checkpoint-{N_STEPS}"
RUN_NAME = 'test_freeze'
LOG_DIR = f'tests/.cache/results/{RUN_NAME}'
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
    'run_name': RUN_NAME,
}


def test_freeze() -> None:

    if os.path.exists(LOG_DIR):
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)

    for architecture, training_style in get_all_architectures_with_all_train_styles():
        print(
            'Testing freeze for {architecture}-architecture '
            f'and {training_style}-training'
        )

        for freeze in [
            'embedder',
            'decoder',
            'unembedder'
            ]:
            trainer_frozen = run_train_process(
                config={
                    **TRAIN_CONFIG,
                    'architecture': architecture,
                    'training_style': training_style,
                    f'freeze_{freeze}': 'True',
                }
            )
            model = trainer_frozen.model

            if freeze == 'embedder':
                state_dict = dict(model.embedder.state_dict())

            elif freeze == 'decoder':
                state_dict = dict(model.decoder.state_dict())

            elif freeze == 'unembedder': 
                state_dict = dict(model.unembedder.state_dict())

            else:
                raise ValueError(
                    f'Unknown freeze type: {freeze}'
                )

            for step in np.arange(1, N_STEPS+1):
                model.from_pretrained(
                    f'{LOG_DIR}/checkpoint-{step}/pytorch_model.bin'
                )

                if freeze == 'decoder':
                    assert all(
                        k in model.decoder.state_dict().keys()
                        for k in state_dict.keys()
                    ), 'keys of original and loaded decoder model mismtach.'

                    for key in state_dict.keys():
                        assert torch.equal(
                            state_dict[key],
                            model.decoder.state_dict()[key]
                        ), f'values for decoder-{key} mismatch'

                elif freeze == 'embedder':
                    assert all(
                        k in model.embedder.state_dict().keys()
                        for k in state_dict.keys()
                    ), 'keys of original and loaded emnbedder model mismtach.'

                    for key in state_dict.keys():
                        assert torch.equal(
                            state_dict[key],
                            model.embedder.state_dict()[key]
                        ), f'values for embedder-{key} mismatch'

                elif freeze == 'unembedder':
                    assert all(
                        k in model.unembedder.state_dict().keys()
                        for k in state_dict.keys()
                    ), 'keys of original and loaded nemebdder model mismtach.'

                    for key in state_dict.keys():
                        assert torch.equal(
                            state_dict[key],
                            model.unembedder.state_dict()[key]
                        ), f'values for unembedder-{key} mismatch'

                else:
                    raise ValueError(
                        f'Unknown freeze type: {freeze}'
                    )

            print(
                f"Removing {LOG_DIR}..."
            )
            shutil.rmtree(LOG_DIR)

if __name__ == "__main__":
    run_tests()