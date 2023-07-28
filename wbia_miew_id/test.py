from datasets import MiewIdDataset, get_train_transforms, get_valid_transforms, get_test_transforms
from logging_utils import WandbContext
from models import MiewIdNet
from etl import preprocess_data, print_basic_stats
from engine import eval_fn, group_eval
from helpers import get_config
from visualization import render_query_results

import os
import torch
import random
import numpy as np

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

    parser.add_argument('--visualize', '--vis', action='store_true')

    return parser.parse_args()

def run_test(config, visualize=False):
    
    checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}/{config.model_params.model_name}-{config.data.image_size[0]}-{config.engine.loss_module}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print('Checkpoints will be saved at: ', checkpoint_dir)


    def set_seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    set_seed_torch(config.engine.seed)

    df_test = preprocess_data(config.data.test.anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.test.n_filter_min, 
                                n_subsample_max=config.data.test.n_subsample_max)
    
    top_names = df_test['name'].value_counts().index[:10]

    df_test = df_test[df_test['name'].isin(top_names)]
    
    test_dataset = MiewIdDataset(
        csv=df_test,
        images_dir=config.data.images_dir,
        transforms=get_test_transforms(config),
        fliplr=config.test.fliplr,
        fliplr_view=config.test.fliplr_view,
        crop_bbox=config.data.crop_bbox
    )
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.engine.valid_batch_size,
        num_workers=config.engine.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(config.engine.device)

    model = MiewIdNet(**dict(config.model_params))
    model.to(device)

    checkpoint_path = config.data.test.checkpoint_path

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
        print('loaded checkpoint from', checkpoint_path)

    test_score, cmc, test_outputs = eval_fn(test_loader, model, device, use_wandb=False, return_outputs=True)




    eval_groups = config.data.test.eval_groups

    if eval_groups:
        grioup_results = group_eval(config, df_test, eval_groups, model)

    if visualize:
        render_query_results(config, model, test_dataset, df_test, test_outputs)

    return test_score

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    config = get_config(config_path)

    visualize = args.visualize

    run_test(config, visualize=visualize)