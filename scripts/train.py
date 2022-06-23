#!/usr/bin/env python3

"""
train.py

Training of models on given data. See get_args() for 
details on command line arguments.

To train a model, multiple core components from ..src/ 
are invoked:

src/batcher: Building PyTorch dataloaders for given data.
src/embedder: Embedding of inputs into embedding space, 
    training-style specific addition of training tokens
    and masking, and computation of training-style specific 
    losses.
    Valid training styles:
        - CSM (Causal Sequence Modeling)
        - BERT (Sequence-BERT)
        - NetBERT (Network-BERT)
        - autoencoder
        - decoding
src/decoder: Model architecture used for decoding / sequence modeling. 
    One of the following:
        - GPT
        - BERT
        - NetBERT
        - autoencoder
        - LinearBaseline
        - PretrainedGPT2 (as provided by HuggingFace)
        - PretrainedBERT (as provided by HuggingFace)
src/unembedder: Projecting sequence output of decoder back 
    to input space.
src/trainer: Trainer for model; invokes instance of 
    Hugging Face's Trainer object.
src/model: Build full model from components (ie., embedder, 
    decoder, unembedder). See make_model() below for details.
"""

import os
import argparse
from typing import Dict
import json
from datetime import datetime
from numpy import random
import pandas as pd
import numpy as np
from torch import manual_seed
import sys
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))
from src.batcher import make_batcher
from src.decoder import make_decoder
from src.embedder import make_embedder
from src.trainer import make_trainer, Trainer
from src.unembedder import make_unembedder
from src.model import Model
from src import tools


def train(config: Dict=None) -> Trainer:
    """Model training according to config.
        -> see get_args() below for all command 
        line arguments.
    """
    
    if config is None:
        config = get_config()

    if config['do_train']:
        os.makedirs(
            config["log_dir"],
            exist_ok=True
        )

        resume_path = str(config["resume_from"]) if config["resume_from"] is not None else None
        
        if resume_path is not None:
            config_filepath = os.path.join(
                config["resume_from"],
                'train_config.json'
            )

            if os.path.isfile(config_filepath):
                print(
                    f'Loading training config from {config_filepath}'
                )

                with open(config_filepath, 'r') as f:
                    config = json.load(f)

            else:

                with open(config_filepath, 'w') as f:
                    json.dump(config, f, indent=2)
            
            checkpoints = [
                int(p.split('checkpoint-')[1])
                for p in os.listdir(resume_path)
                if 'checkpoint-' in p
                and os.path.isdir(os.path.join(resume_path, p))
            ]
            last_checkpoint = max(checkpoints)
            print(
                f'Resuming training from checkpoint-{last_checkpoint} in {resume_path}'
            )
            config["resume_from"] = os.path.join(
                resume_path,
                f'checkpoint-{last_checkpoint}'
            )

        else:
            config_filepath = os.path.join(
                config["log_dir"],
                'train_config.json'
            )
            
            with open(config_filepath, 'w') as f:
                json.dump(config, f, indent=2)

            config["resume_from"] = None

    assert config["training_style"] in {
        'BERT',
        'CSM',
        'NetBERT',
        'autoencoder',
        'decoding'
    }, f'{config["training_style"]} is not supported.'
    
    assert config["architecture"] in {
        'BERT',
        'NetBERT',
        'GPT',
        'autoencoder',
        'PretrainedGPT2',
        'PretrainedBERT',
        'LinearBaseline'
    }, f'{config["architecture"]} is not supported.'
    
    path_tarfile_paths_split = os.path.join(
        config["log_dir"],
        'tarfile_paths_split.json'
    )

    if config['set_seed']:
        random.seed(config["seed"])
        manual_seed(config["seed"])

    if not os.path.isfile(path_tarfile_paths_split): 
        tarfile_paths = tools.data.grab_tarfile_paths(config["data"])
        tarfile_paths_split = tools.data.split_tarfile_paths_train_val(
            tarfile_paths=tarfile_paths,
            frac_val_per_dataset=config["frac_val_per_dataset"],
            n_val_subjects_per_dataset=config["n_val_subjects_per_dataset"],
            n_test_subjects_per_dataset=config["n_test_subjects_per_dataset"],
            n_train_subjects_per_dataset=config["n_train_subjects_per_dataset"],
            seed=config["seed"] if config['set_seed'] else np.random.choice(range(1, 100000))
        )
        print(
            f'Saving tarfile split to {path_tarfile_paths_split}'
        )

        with open(path_tarfile_paths_split, 'w') as f:
            json.dump(tarfile_paths_split, f, indent=2)

    else:
        print(
            f'Loading tarfile split from {path_tarfile_paths_split}'
        )

        with open(path_tarfile_paths_split, 'r') as f:
            tarfile_paths_split = json.load(f)

    train_tarfile_paths = tarfile_paths_split['train']
    validation_tarfile_paths = tarfile_paths_split['validation']
    test_tarfile_paths = tarfile_paths_split['test'] if 'test' in tarfile_paths_split else None
    
    assert all(
        os.path.isfile(f) for f in train_tarfile_paths
    ), f'Some of the training tarfiles in {path_tarfile_paths_split} do not exist.'
    
    assert all(
        os.path.isfile(f) for f in validation_tarfile_paths
    ), f'Some of the validation tarfiles in {path_tarfile_paths_split} do not exist.'
    
    if test_tarfile_paths is not None:
        assert all(
            os.path.isfile(f) for f in test_tarfile_paths
        ), f'Some of the test tarfiles in {path_tarfile_paths_split} do not exist.'

    batcher = make_batcher(
        training_style=config["training_style"],
        sample_random_seq=config["sample_random_seq"],
        seq_min=config["seq_min"],
        seq_max=config["seq_max"],
        bert_seq_gap_min=config["bert_seq_gap_min"],
        bert_seq_gap_max=config["bert_seq_gap_max"],
        decoding_target=config["decoding_target"],
        bold_dummy_mode=config["bold_dummy_mode"]
    )
    train_dataset = batcher.dataset(
        tarfiles=train_tarfile_paths,
        length=config["training_steps"]*config["per_device_training_batch_size"]
    )
    validation_dataset = batcher.dataset(
        tarfiles=validation_tarfile_paths,
        length=config["validation_steps"]*config["per_device_validation_batch_size"]
    )

    if test_tarfile_paths is not None:
        test_dataset = batcher.dataset(
            tarfiles=test_tarfile_paths,
            length=config["test_steps"]*config["per_device_validation_batch_size"]
        )
    
    else:
        test_dataset = None

    def model_init(params: Dict=None):
        model_config = dict(config)

        if params is not None:
            model_config |= params

        return make_model(model_config)

    if config['do_train']:
        tools.configure_wandb(
            config=config,
            entity='athms',
            run_id=config["run_name"],
            project=config["wandb_project_name"],
            mode=config["wandb_mode"]
        )

    trainer = make_trainer(
        model_init=model_init,
        training_style=config["training_style"],
        wandb_mode=config["wandb_mode"],
        run_name=config["run_name"],
        output_dir=config["log_dir"],
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        per_device_train_batch_size=config["per_device_training_batch_size"],
        per_device_eval_batch_size=config["per_device_validation_batch_size"],
        dataloader_num_workers=config["num_workers"],
        optim=config["optim"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_beta_1"],
        adam_beta2=config["adam_beta_1"],
        adam_epsilon=config["adam_epsilon"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        max_steps=config["training_steps"],
        save_steps=config["log_every_n_steps"],
        logging_steps=config["log_every_n_steps"],
        seed=config["seed"] if config['set_seed'] else np.random.choice(range(1, 100000)),
        fp16=config["fp16"],
        deepspeed=config["deepspeed"],
    )

    if config["plot_model_graph"]:
        tools.visualize.plot_model_graph(
            model=trainer.model,
            dataloader=trainer.get_train_dataloader(),
            path=os.path.join(
                config["log_dir"],
                'model_graph'
            )
        )

    if config['do_train']:
        trainer.train(resume_from_checkpoint=config["resume_from"])
        trainer.save_model(
            os.path.join(
                config["log_dir"],
                'model_final'
            )
        )

    if test_dataset is not None:
        test_prediction = trainer.predict(test_dataset)
        pd.DataFrame(
            test_prediction.metrics,
            index=[0]
        ).to_csv(
            os.path.join(
                config["log_dir"],
                'test_metrics.csv'
            ),
            index=False
        )
        np.save(
            os.path.join(
                config["log_dir"],
                'test_predictions.npy'
            ),
            test_prediction.predictions
        )
        np.save(
            os.path.join(
                config["log_dir"],
                'test_label_ids.npy'
            ),
            test_prediction.label_ids
        )

    return trainer


def make_model(model_config: Dict=None):
    """Make model from model_config 
    (as generated by get_config()).
    """
    embedder = make_embedder(
        training_style=model_config["training_style"],
        architecture=model_config["architecture"],
        in_dim=model_config["parcellation_dim"],
        embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_embedding_model"],
        dropout=model_config["dropout"],
        t_r_precision=model_config["tr_precision"],
        max_t_r=model_config["tr_max"],
        masking_rate=model_config["masking_rate"],
        n_positions=model_config["n_positions"]
    )
    decoder = make_decoder(
        architecture=model_config["architecture"],
        num_hidden_layers=model_config["num_hidden_layers"],
        embed_dim=model_config["embedding_dim"],
        num_attention_heads=model_config["num_attention_heads"],
        n_positions=model_config["n_positions"],
        intermediate_dim_factor=model_config["intermediate_dim_factor"],
        hidden_activation=model_config["hidden_activation"],
        dropout=model_config["dropout"],
        autoen_teacher_forcing_ratio=model_config["autoen_teacher_forcing_ratio"],
    )

    if model_config["embedding_dim"] != model_config["parcellation_dim"]:
        unembedder = make_unembedder(
            embed_dim=model_config["embedding_dim"],
            num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
            out_dim=model_config["parcellation_dim"],
            dropout=model_config["dropout"],
        )

    else:
        unembedder = None

    model = Model(
        embedder=embedder,
        decoder=decoder,
        unembedder=unembedder
    )

    if model_config["training_style"] == 'decoding':
        model.switch_decoding_mode(
            is_decoding_mode=True,
            num_decoding_classes=model_config["num_decoding_classes"]
        )

    if model_config["pretrained_model"] is not None:
        model.from_pretrained(model_config["pretrained_model"])

    if model_config["freeze_embedder"]:
        for param in model.embedder.parameters():
            param.requires_grad = False

    if model_config["freeze_decoder"]:
        for param in model.decoder.parameters():
            param.requires_grad = False

    if 'freeze_decoder_without_pooler_heads' in model_config \
        and model_config["freeze_decoder_without_pooler_heads"]:
        for name, param in model.decoder.named_parameters():
            if 'pooler_layer' in name \
            or 'decoding_head' in name \
            or 'is_next_head' in name:
                continue
            else:
                param.requires_grad = False

    if model_config["freeze_unembedder"] and unembedder is not None:
        for param in model.unembedder.parameters():
            param.requires_grad = False

    return model



def get_config(args: argparse.Namespace=None) -> Dict:
    """
    Make config from command line arguments (as created by get_args()).
    Performs additional formating of args required for calling train().
    """

    if args is None:
        args = get_args().parse_args()

    if args.smoke_test == "True":
        args.per_device_training_batch_size =  2
        args.per_device_validation_batch_size = 2
        args.training_steps = 2
        args.validation_steps = 2
        args.test_steps = 2
        args.log_every_n_steps = 1

    if args.num_attention_heads == -1:
        assert (
            args.embedding_dim%64
         ) == 0, f'embedding-dim needs be be multiple of 64 (currently: {args.embedding_dim})' 
        args.num_attention_heads = args.embedding_dim//64

    if args.run_name == 'none':
        args.run_name = f'{args.architecture}'

        if args.architecture != 'LinearBaseline':
            
            if 'Pretrained' not in args.architecture:
                args.run_name += f'_lrs-{args.num_hidden_layers}'

                if args.architecture != 'autoencoder':
                    args.run_name += f'_hds-{args.num_attention_heads}'

            args.run_name += f'_embd-{args.embedding_dim}'
            args.run_name += f'_train-{args.training_style}'
            args.run_name += f'_lr-{str(args.learning_rate).replace(".", "")[1:]}'
            args.run_name += f'_bs-{args.per_device_training_batch_size}'
            args.run_name += f'_drp-{str(args.dropout).replace(".", "")}'

            if args.training_style not in {'decoding', 'autoencoder', 'CSM'}:
                args.run_name += f'_msk-{str(args.masking_rate).replace(".", "")}'

        else:
            args.run_name += f'_train-{args.training_style}'

        args.run_name += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if args.smoke_test == "True":
        args.run_name = f'smoke-test_{args.run_name}'

    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.wandb_mode = args.wandb_mode if args.wandb_mode in {'online', 'offline'} and args.local_rank in {-1, 0} else "disabled"
    
    config = vars(args)

    for arg in config:
        
        if config[arg] in {'True', 'False'}:
            config[arg] = config[arg] == 'True'
        
        elif config[arg] == 'none':
            config[arg] = None

        elif 'subjects_per_dataset' in arg:
            config[arg] = None if config[arg] == -1 else config[arg]

    return config


def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(
        description='run model training'
    )

    # Data pipeline settings:
    parser.add_argument(
        '--data',
        metavar='DIR',
        default='data/upstream',
        type=str,
        help='path to training data directory '
             '(default: data/upstream)'
    )
    parser.add_argument(
        '--frac-val-per-dataset',
        metavar='FLOAT',
        default=0.05,
        type=float,
        help='fraction of fMRI runs per dataset that '
             'are randomly selected as validation data '
             '(default: 0.05)'
    )
    parser.add_argument(
        '--n-val-subjects-per-dataset',
        metavar='INT',
        default=-1,
        type=int,
        help='number of subjects per dataset that are '
             'randomly selected as validation data. '
             '! overrides --frac-val-per-dataset and '
             'requires setting --n-train-subjects-per-dataset' 
    )
    parser.add_argument(
        '--n-test-subjects-per-dataset',
        metavar='INT',
        default=-1,
        type=int,
        help='number of subjects per dataset that are '
             'randomly selected as test data. '
             '! Test set is only created if this is set != -1'
    )
    parser.add_argument(
        '--n-train-subjects-per-dataset',
        metavar='INT',
        default=-1,
        type=int,
        help='number of subjects per dataset that are '
             'randomly selected as training data. '
             '! overrides --frac-val-per-dataset and '
             'requires setting --n-val-subjects-per-dataset' 
    )
    parser.add_argument(
        '--parcellation-dim',
        metavar='INT',
        default=1024,
        type=int,
        help='dimension of input data parcellation (default: 1024). '
             '! This is fixed for the current up-/downstream data.'
    )
    parser.add_argument(
        '--pretrained-model',
        metavar='DIR',
        type=str,
        default='none',
        help='checkpoint used to initialize model weights '
             '(default: none)'
    )


    # Embedder settings:    
    parser.add_argument(
        '--embedding-dim',
        metavar='INT',
        default=768,
        type=int,
        help='dimension of input embedding '
             '(default: 768)'
    )
    parser.add_argument(
        '--num-hidden-layers-embedding-model',
        metavar='INT',
        default=1,
        type=int,
        help='numer of layers of linear embedding model '
             '(default: 1)'
    )
    parser.add_argument(
        '--tr-max',
        metavar='INT',
        default=300,
        type=int,
        help='maximum number of TRs in TR-embeddings '
             '(in seconds; default: 300)'
    )
    parser.add_argument(
        '--tr-precision',
        metavar='FLOAT',
        default=0.2,
        type=float,
        help='precision (ie., frequency) of TR embeddings '
             '(in seconds; default: 0.2). '
             'When set to 0.2, embeddings are created for: '
             '0, 0.2, 0.4, ..., tr-max'
    )
    parser.add_argument(
        '--freeze-embedder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze embedder weights during training '
             '(default: False) '
    )


    # UnEmbedder settings:
    parser.add_argument(
        '--num-hidden-layers-unembedding-model',
        metavar='INT',
        default=1,
        type=int,
        help='numer of hidden layers for linear unembedding model '
             '(default: 1)'
    )
    parser.add_argument(
        '--freeze-unembedder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze unembedder weights during training '
             '(default: False) '
    )


    # Decoder settings:
    parser.add_argument(
        '--architecture',
        metavar='STR',
        default='GPT',
        choices=(
            'BERT',
            'GPT',
            'autoencoder',
            'NetBERT',
            'LinearBaseline'
        ),
        type=str,
        help='Model architecture used for sequence modeling / decoding. '
             'One of {BERT, NetBERT, GPT, autoencoder, LinearBaseline} '
             '(default: GPT) '
    )
    parser.add_argument(
        '--num-hidden-layers',
        metavar='INT',
        default=4,
        type=int,
        help='number of hidden model layers in --architecture '
             '(default: 4). '
             '! Does not apply to LinearBaseline; '
             '! Same number of hidden layers is used for decoder / encoder '
             'parts of autoencoder (ie., default creates encoder and decoder '
             'with 4 hidden layers each)'
    )
    parser.add_argument(
        '--num-attention-heads',
        metavar='INT',
        default=-1,
        type=int,
        help='number of attention heads per transformer layer '
             '(default: embedding-dim // 64). '
             '! Does not apply to non-transformer models'
    )
    parser.add_argument(
        '--intermediate-dim-factor',
        metavar='INT',
        default=4,
        type=int,
        help='scales feed-forward transformer layer dimension relative to '
             'embedding-dim: intermediate-dim-factor * embedding-dim '
             '(default: 4)'
    )
    parser.add_argument(
        '--hidden-activation',
        metavar='STR',
        default='gelu_new',
        choices=(
            'gelu',
            'gelu_new',
            'relu',
            'silu'
        ),
        type=str,
        help='type of hidden activation of transformer layers '
             '(default: gelu_new); '
             'one of {"gelu", "gelu_new", "relu", "silu"}. '
             '! Does not apply to non-transformer models'
    )
    parser.add_argument(
        '--n-positions',
        metavar='INT',
        default=512,
        type=int,
        help='maximum sequence length that transformer model might ever be used with '
             '(default: 512)'
    )
    parser.add_argument(
        '--freeze-decoder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze decoder model weights during training '
             'as specified by --architecture '
             '(default: False) '
    )
    parser.add_argument(
        '--freeze-decoder-without-pooler-heads',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze decoder model weights during training '
             'as specified by --architecture, without pooler layer and '
             ' is-next-pred / decoding heads '
             '(default: False) '
    )

    

    # Trainer settings:
    parser.add_argument(
        '--resume-from',
        metavar='DIR',
        type=str,
        default='none',
        help='continue training from specified checkpoint '
             '(default: none)'
    )
    parser.add_argument(
        '--training-style',
        metavar='STR',
        default='CSM',
        choices=(
            'CSM',
            'BERT',
            'NetBERT',
            'autoencoder',
            'decoding'
        ),
        type=str,
        help='training framework / style (default: CSM); '
             'one of {BERT, CSM, NetBERT, autoencoder, decoding}'
    )
    parser.add_argument(
        '--decoding-target',
        metavar='STR',
        default='none',
        type=str,
        help='key for decoding target variable in .tar-files in --data'
             '(default: none). '
             '! Must be specified when setting --training-style to "decoding"'
    )
    parser.add_argument(
        '--num-decoding-classes',
        metavar='INT',
        default=0,
        type=int,
        help='number of decoding classes (ie., mental states) in --data '
             '(default: 0). '
             '! Must be specified when setting --training-style to "decoding"'
    )
    parser.add_argument(
        '--training-steps',
        metavar='INT',
        default=400000,
        type=int,
        help='number of training steps to perform '
             '(default: 400000)'
    )
    parser.add_argument(
        '--validation-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='number of validation steps to perform at evaluation time '
             '(default: 1000)'
    )
    parser.add_argument(
        '--test-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='number of test steps to perform at test time'
             '(default: 2000). '
             '! Test evaluation only performed if test set created by '
             'setting --n-test-subjects-per-dataset != -1'
    )
    parser.add_argument(
        '--per-device-training-batch-size',
        metavar='INT',
        default=64,
        type=int,
        help='batch size during training per training device '
             '(default: 64)'
    )
    parser.add_argument(
        '--per-device-validation-batch-size',
        metavar='INT',
        default=64,
        type=int,
        help='batch size during evaluation per training device '
             '(default: 64)'
    )
    parser.add_argument(
        '--optim',
        metavar='STR',
        default='adamw_hf',
        type=str,
        help='optimizer to use for training '
             '(default: adamw_hf) -> adamw from HuggingFrace transformer library. '
             'For other options see Huggingface TrainerArgs.'
    )
    parser.add_argument(
        '--learning-rate',
        metavar='FLOAT',
        default=1e-4,
        type=float,
        help='maximum learning rate during training '
             '(default: 1e-4)'
    )
    parser.add_argument(
        '--warmup-ratio',
        metavar='FLOAT',
        default=0.01,
        type=float,
        help='warm-up steps for linear learning rate scheduler '
             'specified as fraction of --training-steps '
             '(default: 0.01)'
    )
    parser.add_argument(
        '--weight-decay',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='weight decay strength (indicating l2-regularisation strength) '
             '(default: 0.1)'
    )
    parser.add_argument(
        '--adam-beta-1',
        metavar='FLOAT',
        default=0.9,
        type=float,
        help='adam beta 1 (default: 0.9)'
    )
    parser.add_argument(
        '--adam-beta-2',
        metavar='FLOAT',
        default=0.999,
        type=float,
        help='adam beta 2 (default: 0.999)'
    )
    parser.add_argument(
        '--adam-epsilon',
        metavar='FLOAT',
        default=1e-8,
        type=float,
        help='adam beta 2 (default: 1e-8)'
    )
    parser.add_argument(
        '--max-grad-norm',
        metavar='FLOAT',
        default=1.0,
        type=float,
        help='maximum gradient clipping norm (default: 1.0)'
    )
    parser.add_argument(
        '--lr-scheduler',
        metavar='STR',
        default='linear',
        choices=(
            'linear',
            'constant_with_warmup',
            'none'
        ),
        type=str,
        help='learning rate scheduler; '
             'one of {linear, constant_with_warmup, none} '
             '(default: linear)'
    )
    parser.add_argument(
        '--sample-random-seq',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        help='whether or not to randomly sample input sequences '
             'from BOLD --data during training '
             '(default: True). '
             'Range for randomly sampled sequence lengths specified by '
             '--seq-min and --seq-max'
    )
    parser.add_argument(
        '--seq-min',
        metavar='INT',
        default=10,
        type=int,
        help='minimum length of randomly sampled BOLD input sequences '
             '(in number of TRs; default: 10)'
    )
    parser.add_argument(
        '--seq-max',
        metavar='INT',
        default=50,
        type=int,
        help='maximum length of randomly sampled BOLD input sequences '
             '(in number of TRs; default: 50)'
    )
    parser.add_argument(
        '--bert-seq-gap-min',
        metavar='INT',
        default=1,
        type=int,
        help='minimum TR gap between two input sequences for BERT-style training '
             '(default: 1). '
             'Gap is randomly sampled between --bert-seq-gap-min and --bert-seq-gap-max'
    )
    parser.add_argument(
        '--bert-seq-gap-max',
        metavar='INT',
        default=5,
        type=int,
        help='maximum TR gap between two input sequences for BERT-style training '
             '(default: 5). '
             'Gap is randomly sampled between --bert-seq-gap-min and --bert-seq-gap-max'
    )
    parser.add_argument(
        '--masking-rate',
        metavar='FLOAT',
        default=0.2,
        type=float,
        help='masking rate for BERT-style training '
             '(default: 0.15)'
    )
    parser.add_argument(
        '--dropout',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='dropout ratio for hidden layers of embedder and decoder model parts '
             '(default: 0.1)'
    )
    parser.add_argument(
        '--autoen-teacher-forcing-ratio',
        metavar='FLAOT',
        default=0.5,
        type=float,
        help='teacher forcing ratio for autoencoder training '
             '(default: 0.5)'
    )

    
    # Logging settings:
    parser.add_argument(
        '--log-dir',
        metavar='DIR',
        type=str,
        default='results/models/upstream',
        help='path where training is logged '
             '(default: results/models/upstream)'
    )
    parser.add_argument(
        '--log-every-n-steps',
        metavar='INT',
        default=10000,
        type=int,
        help='frequence of logging in training steps '
             '(default: 10000)'
    )
    parser.add_argument(
        '--run-name',
        metavar='STR',
        type=str,
        default='none',
        help='descriptor of the training run used for logging and wandb; '
             '! if set to "none", a unique identifier is automatically created'
    )
    parser.add_argument(
        '--wandb-mode',
        metavar='STR',
        choices=(
            'online',
            'offline',
            'disabled'
        ),
        default='disabled',
        help='track training w/ wandb online or offline or not at all '
             '(default: disabled) '
             '! requires setting up weights-and-bias for this machine; '
             'see: https://docs.wandb.ai/'
    )
    parser.add_argument(
        '--wandb-project-name',
        metavar='STR',
        type=str,
        default='learning-from-brains',
        help='name of wandb project where data is logged '
             '(default: learning-from-brains)'
    )
    

    # Other settings:
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=1234,
        type=int,
        help='random seed (default: 1234)'
    )
    parser.add_argument(
        '--set-seed',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        type=str,
        help='whether or not to set random seed (default: True)'
    )
    parser.add_argument(
        '--fp16',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        help='whether or not to use 16-bit precision GPU training '
             '(default: True)'
    )
    parser.add_argument(
        '--deepspeed',
        metavar='DIR',
        default="none",
        type=str,
        help='location of deepspeed configuration file; '
             'automatically adds deepspeed functionality to training if specified '
             '(default: none)'
    )
    parser.add_argument(
        '--local_rank',
        metavar='INT',
        default=-1,
        type=int,
        help='Rank of the process during distributed training '
             '(default: -1)'
    )
    parser.add_argument(
        '--num-workers',
        metavar='INT',
        default=0,
        type=int,
        help='number of data loading workers '
             '(default: 0 -> load in main process)'
    )
    parser.add_argument(
        '--plot-model-graph',
        metavar='BOOL',
        default="False",
        type=str,
        choices=('True', 'False'),
        help='whether or not to save an image of the model graph to log-dir '
             '(default: False)'
    )
    parser.add_argument(
        '--smoke-test',
        metavar='BOOL',
        default="False",
        type=str,
        choices=("True", "False"),
        help='whetehr or not to run training in smoke test-mode '
             '(default: False)'
             'If set to "True", training is restricted by setting: '
             '--per-device-training_batch_size 2 '
             '--per-device-validation_batch_size 2 '
             '--training-steps 2 '
             '--validation-steps 2 '
             '--test-steps 2 '
             '--log-every-n-steps 1'
    )
    parser.add_argument(
        '--bold-dummy-mode',
        metavar='BOOL',
        default='False',
        type=str,
        choices=('True', 'False'),
        help='whether or not to replace BOLD with dummy during training; '
             'for internal testing purposes only! '
             '(default: False)'
    )
    parser.add_argument(
        '--do-train',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether or not to run training '
             '(default: True). '
             'If "False", train() still returns trainer'
    )

    return parser


if __name__ == '__main__':

    trainer = train()