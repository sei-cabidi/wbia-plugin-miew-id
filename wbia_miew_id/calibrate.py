from datasets import MiewIdDataset, get_valid_transforms
from models import MiewIdNet, MiewIdNetTS
from etl import preprocess_data
from engine import calibrate_fn
from helpers import get_config

import os
import torch
import random
import numpy as np

import argparse

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Turn off SSL verify
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )

    parser.add_argument('--visualize', '--vis', action='store_true')

    return parser.parse_args()

def run_calibrate(config, visualize=False):
    
    def set_seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    set_seed_torch(config.engine.seed)

    df_val = preprocess_data(config.data.val.anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.val.n_filter_min, 
                                n_subsample_max=config.data.val.n_subsample_max,
                                use_full_image_path=config.data.use_full_image_path,
                                images_dir = config.data.images_dir)

    # top_names = df_test['name'].value_counts().index[:10]
    # df_test = df_test[df_test['name'].isin(top_names)]

    valid_dataset = MiewIdDataset(
        csv=df_val,
        transforms=get_valid_transforms(config),
        fliplr=config.test.fliplr,
        fliplr_view=config.test.fliplr_view,
        crop_bbox=config.data.crop_bbox,
    )
        
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.engine.valid_batch_size,
        num_workers=config.engine.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(config.engine.device)

    if config.calibration.method == "temperature_scaling":
        model_class = MiewIdNetTS
    else:
        model_class = MiewIdNet

    checkpoint_path = config.data.test.checkpoint_path

    if checkpoint_path:
        weights = torch.load(config.data.test.checkpoint_path, map_location=torch.device(config.engine.device))
        n_train_classes = weights['final.kernel'].shape[-1]
        if config.model_params.n_classes != n_train_classes:
            print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) which is different from actual n_train_classes in the checkpoint -  ({n_train_classes}).")
            config.model_params.n_classes = n_train_classes

        
        model = model_class(**dict(config.model_params))
        model.to(device)

        model.load_state_dict(weights, strict=False)
        print('loaded checkpoint from', checkpoint_path)
        
    else:
        model = model_class(**dict(config.model_params))
        model.to(device)

    calibrate_fn(valid_loader, model, device, checkpoint_path, use_wandb=False) 

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    config = get_config(config_path)

    visualize = args.visualize

    run_calibrate(config, visualize=visualize)