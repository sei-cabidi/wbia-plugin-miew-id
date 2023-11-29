import pandas as pd
import torch
import os
import numpy as np
import random

from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from metrics import compute_distance_matrix, topk_average_precision, precision_at_k
from etl import preprocess_data
from engine import group_eval, eval_fn
from datasets import MiewIdDataset, get_test_transforms
from models import MiewIdNet
from helpers import get_config

from zeno_client import ZenoClient, ZenoMetric


def set_seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ----------------------------- #
# Stage 1: Zeno Setup           #
# ----------------------------- #
# Make sure to set your Zeno API Key at the command
# line using: export ZENOAPIKEY="your api key here"
try:
    client = ZenoClient(os.environ['ZENOAPIKEY'])
except:
    print("You need to add your ZENO API key at the command line: export ZENOAPIKEY='your key here'")

# ----------------------------- #
# Stage 2: WildMe Setup         #
# ----------------------------- #
# Load config
config_path = "./runs/ms-flukebook/msf7-beluga-only-swa/msf7-beluga-only-swa.yaml"
config = get_config(config_path)

# Enforce deterministic behavior
set_seed_torch(config.engine.seed)

# Set device
device = torch.device(config.engine.device)

# Load dataset
df_test = preprocess_data(config.data.test.anno_path,
                          name_keys=config.data.name_keys,
                          convert_names_to_ids=True,
                          viewpoint_list=config.data.viewpoint_list,
                          n_filter_min=config.data.test.n_filter_min,
                          n_subsample_max=config.data.test.n_subsample_max,
                          use_full_image_path=config.data.use_full_image_path,
                          images_dir=config.data.images_dir
                          )

n_train_classes = df_test['name'].nunique()

crop_bbox = config.data.crop_bbox
checkpoint_path = config.data.test.checkpoint_path

test_dataset = MiewIdDataset(
    csv=df_test,
    transforms=get_test_transforms(config),
    fliplr=config.test.fliplr,
    fliplr_view=config.test.fliplr_view,
    crop_bbox=config.data.crop_bbox,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config.engine.valid_batch_size,
    num_workers=config.engine.num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

# Load model
checkpoint_path = config.data.test.checkpoint_path
if checkpoint_path:
    weights = torch.load(config.data.test.checkpoint_path,
                         map_location=torch.device(config.engine.device))
    n_train_classes = weights[list(weights.keys())[-1]].shape[-1]
    print(f"n_train_classes: {n_train_classes}")
    if config.model_params.n_classes != n_train_classes:
        print(
            f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) which is different from actual n_train_classes in the checkpoint -  ({n_train_classes}).")
        config.model_params.n_classes = n_train_classes

    model = MiewIdNet(**dict(config.model_params))
    model.to(device)

    model.load_state_dict(weights)
    print('loaded checkpoint from', checkpoint_path)

else:
    model = MiewIdNet(**dict(config.model_params))
    model.to(device)

model.eval()

eval_groups = config.data.test.eval_groups

if eval_groups:
    df_test = preprocess_data(config.data.test.anno_path,
                              name_keys=config.data.name_keys,
                              convert_names_to_ids=True,
                              viewpoint_list=config.data.viewpoint_list,
                              n_filter_min=None,
                              n_subsample_max=None,
                              use_full_image_path=config.data.use_full_image_path,
                              images_dir=config.data.images_dir)
    group_results = group_eval(config, df_test, eval_groups, model)

# ----------------------------- #
# Stage 3: Output Generation    #
# ----------------------------- #
tk0 = tqdm(test_loader, total=len(test_loader))

# NOTE! Set model in training mode to extract
# class predictions as well as embeddings
model.train()

# Iterate through dataloader and collect inputs and outputs
test_score, cmc, test_outputs = eval_fn(
    test_loader, model, device, use_wandb=False, return_outputs=True)
embeddings, q_pids, distmat = test_outputs

# ----------------------------- #
# Stage 4: Data Processing      #
# ----------------------------- #
k = 5
ranks = list(range(1, k+1))
score, match_mat, topk_idx, topk_names = precision_at_k(
    q_pids, distmat, ranks=ranks, return_matches=True)
match_results = (q_pids, topk_idx, topk_names, match_mat)
print(f"Getting top-{k} results for each image...")

print("Drawing reliability diagram...")

# ----------------------------- #
# Stage 5: Zeno-ification       #
# ----------------------------- #
# TODO: Fix this
df = pd.DataFrame(
    {
        "output": outputs,
        "label": q_pids
    },
    index=image_indices
)

# Explicitly save the index as a column to upload.
df["id"] = df.index

# Create a project
project = client.create_project(
    name="WildMe MiewID",
    view="animal-classification",
    metrics=[
        ZenoMetric(name="accuracy", type="mean", columns=["correct"]),
    ]
)

project.upload_dataset(
    df, id_column="id", data_column='output', label_column="label")

df_system = pd.DataFrame(
    {
        "output": outputs,
    }
)

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

# Measure accuracy for each instance, which is averaged by the ZenoMetric above.
df_system["correct"] = (df_system["output"] == df["target"]).astype(int)

project.upload_system(df_system, name="System A",
                      id_column="id", output_column="output")
