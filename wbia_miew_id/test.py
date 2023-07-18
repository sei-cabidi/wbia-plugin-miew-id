from datasets import MiewIdDataset, get_train_transforms, get_valid_transforms, get_test_transforms
from logging_utils import WandbContext
from models import MiewIdNet
from etl import preprocess_data, print_basic_stats
from losses import fetch_loss
from schedulers import MiewIdScheduler
from engine import eval_fn
from helpers import get_config


import os
import torch
import random
import numpy as np
from dotenv import load_dotenv

import argparse


## NOTE new imports
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from visualization import draw_batch

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

    parser.add_argument('--visualize', action='store_true')

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

    ## TODO modify config.py and switch to test_* variables
    
    df_test = preprocess_data(config.data.val.anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.val.n_filter_min, 
                                n_subsample_max=config.data.val.n_subsample_max)
    
    # print_basic_stats(df_test, individual_key='name_orig')
    
    # n_train_classes = df_train['name'].nunique()

    test_dataset = MiewIdDataset(
        csv=df_test,
        images_dir=config.data.images_dir,
        transforms=get_valid_transforms(config),
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


    # if config.model_params.n_classes != n_train_classes:
    #     print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) which is different from actual n_train_classes ({n_train_classes}). This parameters has to be readjusted in config for proper checkpoint loading after training.")
    #     config.model_params.n_classes = n_train_classes
    
    ## TODO lazyload the final layer
    ## NOTE config.model_params.n_train_classes
    model = MiewIdNet(**dict(config.model_params))
    model.to(device)

    checkpoint_path = config.data.test.checkpoint_path
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
        print('loaded checkpoint from', checkpoint_path)

    ####
    # criterion = fetch_loss()
    # criterion.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.scheduler_params.lr_start)

    # scheduler = MiewIdScheduler(optimizer,**dict(config.scheduler_params))

    # with WandbContext(config):
    #     best_score = run_fn(config, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb)
    ####

    test_score, test_outputs = eval_fn(test_loader, model, device, use_wandb=False, return_outputs=True)

    ###
    # if visualize:

    embeddings, q_pids, distmat = test_outputs

    ###
    k = 5
    output = torch.Tensor(distmat[:, :]) * -1
    y = torch.Tensor(q_pids[:]).squeeze(0)
    ids_tensor = torch.Tensor(q_pids)
    _topk = output.topk(k+1)[1][:, 1:]#.unsqueeze(1)
    topk = ids_tensor[_topk]

    match_mat = topk == y[:, None].expand(topk.shape)
    rank_mat = match_mat.any(axis=1)



    ###
    class IdxSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)
        
    ###
    import cv2

    def stack_images(images, descriptions, match_mask, text_color=(0, 0, 0)):  # OpenCV uses BGR
        assert len(images) == len(descriptions) == len(match_mask), "Number of images, descriptions and match_mask must be the same."

        result_images = []
        for img, desc, match_correct in zip(images, descriptions, match_mask):

            desc_qry, desc_db = desc
            img = (img * 255).astype(np.uint8)
            color = (0, 255, 0) if match_correct else (0, 0, 255)  # green for correct, red for incorrect
            bw = 12
            img = cv2.copyMakeBorder(img, bw, bw, bw, bw, cv2.BORDER_CONSTANT, value=color)

            (tw, th), _ = cv2.getTextSize(desc_qry, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
            
            text_img = np.ones((int(th * 2), img.shape[1], 3), dtype=np.uint8) * 255  # Change height as needed
            
            cv2.putText(text_img, desc_qry, (th, int(th * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4)
            cv2.putText(text_img, desc_db, (img.shape[1]//2 + th, int(th * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4)
            result_images.extend([text_img, img])

        result = np.vstack(result_images)

        return result


    print("Generating visualizations...")
    ## NOTE make random?
    for i in tqdm(range(len(q_pids))):

        #
        vis_idx = _topk[i].tolist()
        vis_idx

        vis_names = topk[i].tolist()
        vis_match_mask = match_mat[i].tolist()
        vis_idx, vis_names
        #

        idxSampler = IdxSampler([i] + vis_idx)

        vis_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config.engine.valid_batch_size,
                num_workers=config.engine.num_workers,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                sampler = idxSampler
            )
        
        
        ## NOTE make render_transformed configurable
        batch_images = draw_batch(
            config, vis_loader,  model, images_dir = 'dev_test', method='gradcam_plus_plus', eigen_smooth=False, 
            render_transformed=True, show=False)


        ##
        df_vis = df_test.iloc[vis_idx]

        viewpoints = df_vis['viewpoint'].values
        names = df_vis['name'].values
        indices = df_vis.index.values

        qry_row = df_test.iloc[i]
        qry_name = qry_row['name']
        qry_viewpoint = qry_row['viewpoint']
        qry_idx = i

        desc_qry = [f"Query: {qry_name} {qry_viewpoint} ({qry_idx})" for i in range(len(viewpoints))]
        desc_db = [f"Match: {name} {viewpoint} ({idx})" for name, viewpoint, idx in zip(names, viewpoints, indices)]

        descriptions = [(q, d) for q, d in zip(desc_qry, desc_db)]

        ##

        vis_result = stack_images(batch_images, descriptions, vis_match_mask)

        output_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}/visualizations"
        os.makedirs(output_dir, exist_ok=True)

        output_name = f"vis_{qry_name}_{qry_viewpoint}_{qry_idx}_top{k}.png"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, vis_result)

        print(f"Saved visualization to {output_path}")



    return test_score

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    config = get_config(config_path)

    visualize = args.visualize

    run_test(config, visualize=visualize)