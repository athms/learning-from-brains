#!/usr/bin/env python3 

import os
from datetime import datetime


def configure_wandb(
    config,
    entity: str=None,
    project: str='learning-from-brains',
    mode: str='online',
    set_wandb_run_id: bool=False,
    run_id: str=None
    ) -> None:
    
    if mode in {"online", 'offline'}: 
        
        os.environ['WANDB_USERNAME'] = entity
        os.environ['WANDB_PROJECT'] = project
        os.environ['WANDB_MODE'] = mode
        os.environ['WANDB_LOG_MODEL'] = 'true'
        os.environ['WANDB_WATCH'] = 'all'
        os.environ['WANDB_DIR'] = config["log_dir"]

        if set_wandb_run_id:

            if run_id is None:
                run_id = f'model-{config["architecture"]}'
                run_id += f'_training-{config["training_style"]}'
                run_id += f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
            
            os.environ['WANDB_RUN_ID'] = run_id
        
    elif mode in {'none', 'disabled'}:
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE'] = 'disabled'

    else:
        raise ValueError(f'unknown wandb_model: {mode}.')