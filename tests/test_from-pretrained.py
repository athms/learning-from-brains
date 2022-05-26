#!/usr/bin/env python3

import os
import shutil
import torch
from tests import get_all_architectures_with_all_train_styles,\
    run_tests,\
    run_train_process,\
    DEFAULT_CONFIG


N_STEPS = 10
LAST_CHECKPOINT = f"checkpoint-{N_STEPS}"
RUN_NAME = 'test_from-pretrained'
LOG_DIR = f'tests/.cache/results/{RUN_NAME}'
TRAIN_CONFIG = {
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
    "decoding_target": "task_label", 
    "num_decoding_classes": 26, 
    "fp16": False,
    "log_dir": LOG_DIR,
    'run_name': RUN_NAME,
}

def test_from_pretrained() -> None:

    if os.path.exists(LOG_DIR):
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)

    for architecture, training_style in get_all_architectures_with_all_train_styles():
        print(
            f'Testing from-pretrained for {architecture}-architecture '
            f'and {training_style}-training'
        )

        # pretrain model
        trainer = run_train_process(
            config={
                    **TRAIN_CONFIG,
                    'architecture': architecture,
                    'training_style': training_style,
                }
            )
        model_pretrained = trainer.model

        # reload pretrained model
        trainer_from_pretrained = run_train_process(
            config={
                    **TRAIN_CONFIG,
                    'architecture': architecture,
                    'training_style': training_style,
                    'pretrained_model': f"{LOG_DIR}/model_final/pytorch_model.bin",
                    'do_train': False
                }
            )
        model_from_pretrained = trainer_from_pretrained.model

        # test whether model elements match
        for model_element in ['embedder', 'decoder', 'unembedder']:
            
            if model_element == 'embedder':
                assert all(
                    k in model_from_pretrained.embedder.state_dict().keys()
                    for k in model_pretrained.embedder.state_dict().keys()
                ), 'keys of pretrained and loaded embedder model mismtach.'

                for key in model_pretrained.embedder.state_dict().keys():
                    assert torch.equal(
                        model_pretrained.embedder.state_dict()[key],
                        model_from_pretrained.embedder.state_dict()[key]
                    ), f'values for embedder-{key} mismatch'

            elif model_element == 'decoder':
                assert all(
                    k in model_from_pretrained.decoder.state_dict().keys()
                    for k in model_pretrained.decoder.state_dict().keys()
                ), 'keys of pretrained and loaded decoder model mismtach.'

                for key in model_pretrained.decoder.state_dict().keys():
                    assert torch.equal(
                        model_pretrained.decoder.state_dict()[key],
                        model_from_pretrained.decoder.state_dict()[key]
                    ), f'values for decoder-{key} mismatch'

            elif model_element == 'unembedder':
                assert all(
                    k in model_from_pretrained.unembedder.state_dict().keys()
                    for k in model_pretrained.unembedder.state_dict().keys()
                ), 'keys of pretrained and loaded unembedder model mismtach.'

                for key in model_pretrained.unembedder.state_dict().keys():
                    assert torch.equal(
                        model_pretrained.unembedder.state_dict()[key],
                        model_from_pretrained.unembedder.state_dict()[key]
                    ), f'values for unembedder-{key} mismatch'

            else:
                raise ValueError(
                    f'Unknown model element: {model_element}'
                )

    print(
        f"Removing {LOG_DIR}..."
    )
    shutil.rmtree(LOG_DIR)

if __name__ == "__main__":
    run_tests()