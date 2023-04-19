from datasets import TbdDataset, get_train_transforms, get_valid_transforms
from logging_utils import init_wandb
from models import TbdNet
from etl import preprocess_data, print_intersect_stats
from losses import fetch_loss
from schedulers import TbdScheduler
from engine import run_fn
from helpers import get_config


import os
import torch
import random
import numpy as np
from dotenv import load_dotenv

import argparse

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    return parser.parse_args()

def run(config_path):
    
    config = get_config(config_path)

    checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}/{config.model_params.model_name}-{config.DIM[0]}-{config.loss_module}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print('Checkpoints will be saved at: ', checkpoint_dir)


    def set_seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    set_seed_torch(config.SEED)

    df_train = preprocess_data(config.data.train_anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.train_n_filter_min, 
                                n_subsample_max=config.data.train_n_subsample_max)
    
    df_val = preprocess_data(config.data.val_anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.val_n_filter_min, 
                                n_subsample_max=config.data.val_n_subsample_max)

    print_intersect_stats(df_train, df_val)


    n_train_classes = df_train['name'].nunique()

    train_dataset = TbdDataset(
        csv=df_train,
        images_dir = config.data.images_dir,
        transforms=get_train_transforms(config),
    )
        
    valid_dataset = TbdDataset(
        csv=df_val,
        images_dir=config.data.images_dir,
        transforms=get_valid_transforms(config),
    )
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(config.device)

    if config.model_params.n_classes != n_train_classes:
        print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) which is different from actual n_train_classes ({n_train_classes}). This parameters has to be readjusted in config for proper checkpoint loading after training.")
        config.model_params.n_classes = n_train_classes
    model = TbdNet(**dict(config.model_params))
    model.to(device)

    criterion = fetch_loss()
    criterion.to(device)
        

    optimizer = torch.optim.Adam(model.parameters(), lr=config.scheduler_params.lr_start)

    scheduler = TbdScheduler(optimizer,**dict(config.scheduler_params))


    if config.use_wandb:
        load_dotenv()
        init_wandb(config.exp_name, config.project_name, config=None)

    run_fn(config, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.use_wandb)

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    print(f"Loading config from path: {config_path}")

    run(config_path)