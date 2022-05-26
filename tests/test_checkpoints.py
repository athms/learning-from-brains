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
RUN_NAME = 'test_checkpoints'
LOG_DIR = f'tests/.cache/results/{RUN_NAME}'
TRAIN_CONFIG = {
    **DEFAULT_CONFIG,
    "data": "tests/data/tarfiles",
    "training_steps": N_STEPS,
    "validation_steps": 2,
    "log_every_n_steps": 2,
    "per_device_training_batch_size": 2,
    "per_device_validation_batch_size": 2,
    'num_hidden_layers': 2,
    'num_attention_heads': 2,
    'autoen_teacher_forcing_ratio': 0, # no teacher forcing during training
    "wandb_mode": "disabled",
    "decoding_target": "task_label", # MDTB test data
    "num_decoding_classes": 26, # MDTB test data
    "fp16": False,
    "log_dir": LOG_DIR,
    'run_name': RUN_NAME
}
RESTART_CONFIG = {
    **TRAIN_CONFIG,
    'resume_from': f'{LOG_DIR}',
    'run_name': f'{RUN_NAME}_restart',
    "training_steps": N_STEPS
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_checkpoint_weights() -> None:

    if os.path.exists(LOG_DIR):
        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)
    
    for architecture, training_style in get_all_architectures_with_all_train_styles():
        print(
            f'Testing checkpoint weights for {architecture}-architecture '
            f'and {training_style}-training'
        )
        trainer_after_training = run_train_process(
            config={
                **TRAIN_CONFIG,
                'architecture': architecture,
                'training_style': training_style
            }
        )
        trainer_after_restart = run_train_process(
            config={
                **RESTART_CONFIG,
                'architecture': architecture,
                'training_style': training_style
            }
        )
        model = trainer_after_training.model
        loaded_model = trainer_after_restart.model
        # TODO: identify device used my model
        #loaded_model.to(device)
        assert model.state_dict().keys() == loaded_model.state_dict().keys()
        for key in model.state_dict().keys():
            assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key])

        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)


def test_checkpoint_forward_pass() -> None:
    
    for architecture, training_style in get_all_architectures_with_all_train_styles():
        print(
            f'Testing checkpoint weights for '
            f'{architecture}-architecture and '
            f'{training_style}-training'
        )
        trainer_after_training = run_train_process(
            config={
                **TRAIN_CONFIG,
                'architecture': architecture,
                'training_style': training_style
            }
        )
        trainer_after_restart = run_train_process(
            config={
                **RESTART_CONFIG,
                'architecture': architecture,
                'training_style': training_style
            }
        )
        model = trainer_after_training.model
        loaded_model = trainer_after_restart.model
        train_dataloader = trainer_after_training.get_train_dataloader()
        batch = next(iter(train_dataloader))
        batch = trainer_after_training._move_batch_to_device(batch=batch)
        assert model.state_dict().keys() == loaded_model.state_dict().keys()
        for key in model.state_dict().keys():
            assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key])
        model.eval()
        loaded_model.eval()
        batch_prepped = model.embedder.prep_batch(batch)
        # test embdder forward pass
        inputs_embeds = model.embedder(batch=batch_prepped)
        inputs_embeds_loaded = loaded_model.embedder(batch=batch_prepped)
        assert torch.equal(
            inputs_embeds,
            inputs_embeds_loaded
        ), (
            f"original embedder outputs: {inputs_embeds} "
            f"dtype: {inputs_embeds.dtype}, "
            f"loaded embedder outputs: {inputs_embeds_loaded} dtype:"
            f" {inputs_embeds_loaded.dtype}"
        )
        batch_prepped['inputs_embeds'] = inputs_embeds
        # test decoder forward pass
        outputs_decoder = model.decoder(batch=batch_prepped)
        outputs_decoder_loaded = loaded_model.decoder(batch=batch_prepped)
        
        if training_style == 'decoding':
            # test decoding outputs
            assert torch.equal(
                outputs_decoder["pooler_outputs"],
                outputs_decoder_loaded["pooler_outputs"]
            ), (
                f"original decoder outputs: {outputs_decoder['pooler_outputs']} "
                f"dtype: {outputs_decoder['pooler_outputs'].dtype}, "
                f"loaded decoder outputs: {outputs_decoder_loaded['pooler_outputs']} dtype:"
                f" {outputs_decoder_loaded['pooler_outputs'].dtype}"
            )
            assert torch.equal(
                outputs_decoder["decoding_logits"],
                outputs_decoder_loaded["decoding_logits"]
            ), (
                f"original decoder outputs: {outputs_decoder['decoding_logits']} "
                f"dtype: {outputs_decoder['decoding_logits'].dtype}, "
                f"loaded decoder outputs: {outputs_decoder_loaded['decoding_logits']} dtype:"
                f" {outputs_decoder_loaded['decoding_logits'].dtype}"
            )

        else:
            if architecture == 'autoencoder':
                # test encoder pooler output
                assert torch.equal(
                    outputs_decoder["pooler_outputs"],
                    outputs_decoder_loaded["pooler_outputs"]
                ), (
                    f"original pooler output encoder: {outputs_decoder['pooler_outputs']} "
                    f"dtype: {outputs_decoder['pooler_outputs'].dtype}, "
                    f"loaded pooler output encoder: {outputs_decoder_loaded['pooler_outputs']} dtype:"
                    f" {outputs_decoder_loaded['pooler_outputs'].dtype}"
                )

            else:
                # test decoder output
                assert torch.equal(
                    outputs_decoder["outputs"],
                    outputs_decoder_loaded["outputs"]
                ), (
                    f"original decoder outputs: {outputs_decoder['outputs']} "
                    f"dtype: {outputs_decoder['outputs'].dtype}, "
                    f"loaded decoder outputs: {outputs_decoder_loaded['outputs']} dtype:"
                    f" {outputs_decoder_loaded['outputs'].dtype}"
                )
            # test unembedder forward pass
            outputs_unembedder = model.unembedder(inputs=outputs_decoder['outputs'])
            outputs_unembedder_loaded = loaded_model.unembedder(inputs=outputs_decoder_loaded['outputs'])
            assert torch.equal(
                outputs_unembedder["outputs"],
                outputs_unembedder_loaded["outputs"]
            ), (
                f"original decoder outputs: {outputs_unembedder['outputs']} "
                f"dtype: {outputs_unembedder['outputs'].dtype}, "
                f"loaded decoder outputs: {outputs_unembedder_loaded['outputs']} dtype:"
                f" {outputs_unembedder_loaded['outputs'].dtype}"
            )

        print(
            f"Removing {LOG_DIR}..."
        )
        shutil.rmtree(LOG_DIR)
            

if __name__ == "__main__":

    run_tests()