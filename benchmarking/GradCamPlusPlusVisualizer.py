import os
import pandas as pd
import torch
import typing as tp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
from wbia_miew_id.evaluate import Evaluator
from wbia_miew_id.visualization import render_single_query_result
from wbia_miew_id.visualization import draw_batch
from wbia_miew_id.visualization import SimilarityToConceptTarget
from wbia_miew_id.visualization import generate_embeddings
from wbia_miew_id.datasets.default_dataset import MiewIdDataset
from wbia_miew_id.engine import calculate_matches
from wbia_miew_id.engine import eval_fn, group_eval_run
from wbia_miew_id.datasets import get_test_transforms

from pytorch_grad_cam import GradCAMPlusPlus

from visualizer import Visualizer

class GradCamPlusPlusVisualizer(Visualizer):
    def __init__(
            self,
            root: str | Path,
            checkpoint_path: str | Path,
            output_dir: str | Path,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):
        self.root = root
        self.device = device
        self.output_dir = output_dir
        self.model_params = {
            'model_name': 'efficientnetv2_rw_m',
            'use_fc': False,
            'fc_dim': 2048,
            'dropout': 0,
            'loss_module': 'arcface_subcenter_dynamic',
            's': 51.960399844266306,
            'margin': 0.32841442327915477,
            'pretrained': True,
            'n_classes': 11968,
            'k': 3
        }
        self.evaluator = Evaluator(
            device=self.device,
            seed=0,
            anno_path=str(Path(self.root,'benchmark_splits/test.csv')),
            name_keys=['name'],
            viewpoint_list=None,
            use_full_image_path=True,
            images_dir=None,
            image_size=(440, 440),
            crop_bbox=True,
            valid_batch_size=12,
            num_workers=8,
            eval_groups=[['species', 'viewpoint']],
            fliplr=False,
            fliplr_view=[],
            n_filter_min=2,
            n_subsample_max=10,
            model_params=self.model_params,
            checkpoint_path = self.checkpoint_path,
            model=None,
            visualize=True,
            visualization_output_dir=self.visualization_output_dir
        )

    # Creates a new Pytorch dataloader from two indices
    def create_dataloader(self, d, idx1, idx2):
        # Extract the data from original dataset
        data1 = d[idx1]
        data1['name'] = data1['label']
        data2 = d[idx2]
        data1['name'] = data2['label']

        # Create a new MiewIdDataset object from extracted data
        new_dataset = MiewIdDataset(
            pd.DataFrame.from_dict([data1,data2]),
            transforms=get_test_transforms((self.evaluator.image_size[0], self.evaluator.image_size[1])),
            fliplr=self.evaluator.fliplr,
            fliplr_view=self.evaluator.fliplr_view,
            crop_bbox=self.evaluator.crop_bbox,
        )

        # Create a new DataLoader from the MiewIdDataset
        loader = torch.utils.data.DataLoader(
            new_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        return loader

    def create_dataframe(self, df, idx1, idx2):
        df_dict = df.to_dict('records')
        df_result = pd.DataFrame([df_dict[idx1], df_dict[idx2]])
        return df_result

    def create_config(self):
        return {
            "device":{
                "engine": "cuda"
            }
        }

    def _gradcam(self, image1, image2, model):
        # Create a new dataloader from the two image indices
        loader = self.create_dataloader(dataset, image1, image2)
        model.eval()

        # Generate the embeddings and extract relevant fields
        embeddings, labels, images, paths, bboxes, thetas = generate_embeddings(self.device, self.evaluator.model, loader)

        # Extract features
        image1_features = embeddings.iloc[0].values
        image1_features = torch.Tensor(image1_features).to(self.device)

        image2_features = embeddings.iloc[1].values
        image2_features = torch.Tensor(image2_features).to(self.device)

        similarity1 = SimilarityToConceptTarget(image1_features)
        similarity2 = SimilarityToConceptTarget(image2_features)
        
        image1_ = images[0].unsqueeze(0)
        image2_ = images[1].unsqueeze(0)

        # Create GradCAM++ Object and generate results
        target_layers = model.backbone.conv_head
        generate_cam = GradCAMPlusPlus(model=model,target_layers=[target_layers],use_cuda=True)

        stack_tensor = torch.cat([image1_, image2_])
        stack_target = [similarity1, similarity2]
        results_cam = generate_cam(input_tensor=stack_tensor, targets=stack_target, aug_smooth=False, eigen_smooth=False)
        return results_cam

    def generate(self, image_query, image_match, is_match: bool = False) -> dict:
        raise NotImplementedError
    
if __name__ == "__main__":
     root = Path("/srv/transparency/wildbook_prototype/data/beluga_example_miewid")
     checkpoint_path = Path("/srv/transparency/wildbook_prototype/data/beluga-model-data/beluga-single-species.bin")