#!/usr/bin/env python3

import os
import inspect
import sys
import traceback
import psutil
from unittest.mock import patch
from src.trainer import Trainer
from scripts.train import train, get_args, get_config


DEFAULT_AGRSPARSE = get_args()
DEFAULT_CONFIG = get_config(DEFAULT_AGRSPARSE.parse_args([]))

def get_GPT_with_pretrain_styles():
    
    return [
        ('GPT', 'CSM'),
    ]

def get_BERT_with_pretrain_styles():
    
    return [
        ('BERT', 'BERT'),
        ('NetBERT', 'NetBERT')
    ]

def get_NetBERT_with_pretrain_styles():
    
    return [
        ('NetBERT', 'NetBERT')
    ]

def get_autoencoder_with_pretrain_styles():
    
    return [
        ('autoencoder', 'autoencoder'),
    ]

def get_architectures_with_pretrain_styles():
    
    return (
        get_GPT_with_pretrain_styles() + 
        get_BERT_with_pretrain_styles() +
        get_autoencoder_with_pretrain_styles() +
        get_NetBERT_with_pretrain_styles()
    )

def get_architectures_with_all_train_styles():
    
    return (
        get_GPT_with_pretrain_styles() + 
        get_BERT_with_pretrain_styles() +
        get_autoencoder_with_pretrain_styles() +
        get_NetBERT_with_pretrain_styles() +
        [
            ('GPT', 'decoding'),
            ('BERT', 'decoding'),
            ('autoencoder', 'decoding'),
        ]
    )

def get_pretrained_architectures_with_pretrain_styles():
    return [
        ('PretrainedGPT2', 'CSM'),
        ('PretrainedBERT', 'BERT'),
    ]

def get_pretrained_architectures_with_all_train_styles():
    return (
        get_pretrained_architectures_with_pretrain_styles() +
        [
            ('PretrainedGPT2', 'decoding'),
            ('PretrainedBERT', 'decoding'),
        ]
    )

def get_all_architectures_with_all_train_styles():
    return (
        get_architectures_with_all_train_styles() +
        get_pretrained_architectures_with_all_train_styles()
    )

def get_all_architectures_with_decoding():
    return [
        ('GPT', 'decoding'),
        ('BERT', 'decoding'),
        ('NetBERT', 'decoding'),
        ('autoencoder', 'decoding'),
        ('PretrainedGPT2', 'decoding'),
        ('PretrainedBERT', 'decoding'),
    ]


def launched_by_deepspeed():
    parent = psutil.Process(os.getppid())
    return "deepspeed.launcher.launch" in parent.cmdline()

DEEPSPEED_MODE = launched_by_deepspeed()

def am_first_deepspeed_child():
    if not DEEPSPEED_MODE:
        return False
    parent = psutil.Process(os.getppid())
    children = parent.children()
    return os.getpid() == children[0].pid if children else False


def to_cl_args(config):
    cl_args = []

    for k, v in config.items():

        if v is None:
            v = 'none'
        
        if k != 'local_rank':
            cl_args.extend(
                (
                    f'--{k.replace("_", "-")}',
                    str(v)
                )
            )
        
        else:
            cl_args.extend(
                (
                    f'--{k}',
                    str(v)
                )
            )
    
    return cl_args


def run_train_process(
    config,
    use_deepspeed=DEEPSPEED_MODE
    ) -> Trainer:
    cl_args = [""] + to_cl_args(config)
    
    if not use_deepspeed or am_first_deepspeed_child():
        
        with patch.object(sys, "argv", cl_args):
            trainer = train(config)

    return trainer


def get_test_functions():
    return [
        (name, obj)
        for name, obj in inspect.getmembers(sys.modules["__main__"])
        if (inspect.isfunction(obj) and
        name.startswith("test") and
        obj.__module__ == "__main__")
    ]


def run_tests():
    if DEEPSPEED_MODE and not am_first_deepspeed_child():
       return
    test_functions = get_test_functions()
    passing_tests = []
    failing_tests = []
    assertion_errors = []
    print("Running tests:")
    
    for (name, test_function) in test_functions:
        print("")
        print(name)
        
        try:
            test_function()
            passing_tests.append(name)
        
        except AssertionError as e:
            failing_tests.append(name)
            assertion_errors.append(
                (
                    e,
                    traceback.format_exc()
                )
            )
    
    print("")
    print("Test report:")
    print(
        f"\t{len(passing_tests)} passed, {len(failing_tests) } failed"
    )
    print("")
    print("Failing tests:")
    
    for test, error in zip(failing_tests, assertion_errors):
        print("")
        print(test)
        print(error[1])
        print(error[0])
    
    if failing_tests:
        sys.exit(1)