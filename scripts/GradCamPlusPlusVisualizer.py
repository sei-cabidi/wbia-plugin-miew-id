import cv2
import os
import pandas as pd
import numpy as np
import torch
import typing as tp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm.auto import tqdm

from pathlib import Path
from wbia_miew_id.evaluate import Evaluator
from wbia_miew_id.visualization import render_single_query_result
from wbia_miew_id.visualization import draw_batch
from wbia_miew_id.visualization import SimilarityToConceptTarget
from wbia_miew_id.visualization import generate_embeddings
from wbia_miew_id.visualization import stack_match_images
from wbia_miew_id.visualization import load_image
from wbia_miew_id.visualization import show_cam_on_image
from wbia_miew_id.datasets.default_dataset import MiewIdDataset
from wbia_miew_id.engine import calculate_matches
from wbia_miew_id.engine import eval_fn, group_eval_run
from wbia_miew_id.datasets import get_test_transforms
from wbia_miew_id.metrics import precision_at_k

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM

from .visualizer import Visualizer

from . import Image_Retrieval
from . import BN_Inception

class GradCamPlusPlusVisualizer(Visualizer):
    def __init__(
            self,
            root: str | Path,
            output_dir: str | Path,
            df_test,
            test_dataset,
            match_mat,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint_path: str | Path = Path("/srv/transparency/wildbook_prototype/data/beluga-model-data/beluga-single-species.bin")
        ):
        self.root = root
        self.device = device
        self.output_dir = output_dir
        self.df_test = df_test
        self.test_dataset = test_dataset
        self.match_mat = match_mat
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
            checkpoint_path = checkpoint_path,
            model=None,
            visualize=True,
            visualization_output_dir=output_dir
        )

        self.data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        self.paths = []
        self.bboxes = []

        with torch.no_grad():
            for batch in tqdm(self.data_loader, total=len(self.data_loader)):
                self.paths.extend(batch["file_path"])
                self.bboxes.extend(batch["bbox"])
        
        self.bboxes = [t.int().tolist() for t in self.bboxes]

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

    def _gradcam(self, image1, image2, model, plusplus=True):
        # Create a new dataloader from the two image indices
        loader = self.create_dataloader(self.test_dataset, image1, image2)
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

        # Create GradCAM++/GradCAM Object and generate results
        target_layers = model.backbone.conv_head
        if plusplus:
            generate_cam = GradCAMPlusPlus(model=model,target_layers=[target_layers],use_cuda=True)
        else:
            generate_cam = GradCAM(model=model,target_layers=[target_layers],use_cuda=True)

        stack_tensor = torch.cat([image1_, image2_])
        stack_target = [similarity1, similarity2]
        results_cam = generate_cam(input_tensor=stack_tensor, targets=stack_target, aug_smooth=False, eigen_smooth=False)
        return results_cam

    def _gradcam_on_image(self, image1, image2, model, plusplus=True):
        # Create a new dataloader from the two image indices
        loader = self.create_dataloader(self.test_dataset, image1, image2)
        model.eval()

        # Generate the embeddings and extract relevant fields
        embeddings, labels, images, paths, bboxes, thetas = generate_embeddings(self.device, self.evaluator.model, loader)

        # Extract features
        image1_features = images[0]
        image1_features = torch.Tensor(image1_features).to(self.device)
        image1_features = image1_features.flatten()[:2152]

        image2_features = images[1]
        image2_features = torch.Tensor(image2_features).to(self.device)
        image2_features = image2_features.flatten()[:2152]

        similarity1 = SimilarityToConceptTarget(image1_features)
        similarity2 = SimilarityToConceptTarget(image2_features)
        
        image1_ = images[0].unsqueeze(0)
        image2_ = images[1].unsqueeze(0)

        # Create GradCAM++/GradCAM Object and generate results
        target_layers = model.backbone.conv_head
        if plusplus:
            generate_cam = GradCAMPlusPlus(model=model,target_layers=[target_layers],use_cuda=True)
        else:
            generate_cam = GradCAM(model=model,target_layers=[target_layers],use_cuda=True)

        stack_tensor = torch.cat([image1_, image2_])
        stack_target = [similarity1, similarity2]
        results_cam = generate_cam(input_tensor=stack_tensor, targets=stack_target, aug_smooth=False, eigen_smooth=False)
        return results_cam

    def heatmap(self, results_cam, qry_idx, db_idx):
        qry_image_path = self.paths[qry_idx]
        qry_float = load_image(qry_image_path)
        qry_bbox = self.bboxes[qry_idx]
        x1, y1, w, h = qry_bbox

        qry_float = qry_float[y1 : y1 + h, x1 : x1 + w]
        if min(qry_float.shape) < 1:
            # Use original image
            qry_float = qry_float = load_image(qry_image_path)

        qry_float_norm = (qry_float - qry_float.min()) / (qry_float.max() - qry_float.min())
        db_grayscale_cam_res = cv2.resize(results_cam[0, :], (qry_float_norm.shape[1], qry_float_norm.shape[0]))
        cam_image_qry = show_cam_on_image(qry_float_norm, db_grayscale_cam_res, use_rgb=True)

        db_image_path = self.paths[db_idx]
        db_float = load_image(db_image_path)
        db_bbox = self.bboxes[db_idx]
        x1, y1, w, h = db_bbox
        db_float = db_float[y1 : y1 + h, x1 : x1 + w]
        if min(db_float.shape) < 1:
            # Use original image
            db_float = db_float = load_image(db_image_path)

        db_float_norm = (db_float - db_float.min()) / (db_float.max() - db_float.min())
        qry_grayscale_cam_res = cv2.resize(results_cam[1, :], (db_float_norm.shape[1], db_float_norm.shape[0]))
        cam_image_db = show_cam_on_image(db_float_norm, qry_grayscale_cam_res, use_rgb=True)

        return cam_image_qry, cam_image_db

    def imshow_convert(self, raw):
        '''
            convert the heatmap for imshow
        '''
        heatmap = np.array(cv2.applyColorMap(np.uint8(255*(1.-raw)), cv2.COLORMAP_JET))
        return heatmap

    def point_specific_map(self, path_1, path_2, size=(224,224)):
        eg = Image_Retrieval.Explanation_generator()
        inputs_1, image_1, inputs_2, image_2 = eg.get_input_from_path(path_1, path_2)
        embed_1, map_1, ori_1, fc_1, embed_2, map_2, ori_2, fc_2 = eg.get_embed(inputs_1=inputs_1, inputs_2=inputs_2)

        eg.Decomposition = eg.Overall_map(map_1 = map_1, map_2 = map_2, fc_1 = fc_1, fc_2 = fc_2, mode = 'GMP')

        # query point, position in the feature matrix (not the x,y in image)
        query_point_1 = [100, 128] 
        query_point_2 = [100, 128] 

        # Use stream=1 for query point on image 1, the generated map is for image 2 (partial_2). vice versa
        partial_1 = eg.Point_Specific(decom=eg.Decomposition, point=query_point_2, stream=2)
        partial_2 = eg.Point_Specific(decom=eg.Decomposition, point=query_point_1, stream=1)

        partial_1 = cv2.resize(partial_1, (size[1], size[0]))
        partial_2 = cv2.resize(partial_2, (size[1], size[0]))
        partial_1 = partial_1 / np.max(partial_1)
        partial_2 = partial_2 / np.max(partial_2)

        image_overlay_1 = image_1 * 0.7 + self.imshow_convert(partial_1) / 255.0 * 0.3
        image_overlay_2 = image_2 * 0.7 + self.imshow_convert(partial_2) / 255.0 * 0.3

        # plt.imshow(image_overlay_1)
        # plt.savefig("/home/jwidjaja/test.png")

        # plt.imshow(image_overlay_2)
        # plt.savefig("/home/jwidjaja/test2.png")

        heatmap_1 = self.imshow_convert(partial_1)
        heatmap_2 = self.imshow_convert(partial_2)

        return image_overlay_1, heatmap_1, image_overlay_2, heatmap_2


    def generate(
            self,
            image_query: np.ndarray | torch.Tensor,
            image_match: np.ndarray | torch.Tensor,
            **kwargs
        ) -> dict:
        """For a single query-match pair of images, generate a visualization and associated metadata."""

        loader = self.create_dataloader(self.test_dataset, kwargs['query_idx'], kwargs['match_idx'])

        batch_images_gradcam_plus_plus = draw_batch(self.device, loader, self.evaluator.model, method='gradcam_plus_plus')
        batch_images_gradcam = draw_batch(self.device, loader, self.evaluator.model, method='gradcam')
        descriptions = [(f"Query {kwargs['query_idx']}", f"Match {kwargs['match_idx']}")]
        vis_match_mask = [self.match_mat[kwargs['query_idx']].tolist()[0]]
        print(f"{len(batch_images_gradcam_plus_plus)} {len(descriptions)} {len(vis_match_mask)}")
        vis_result_gradcam_plus_plus = stack_match_images(batch_images_gradcam_plus_plus, descriptions, vis_match_mask)
        vis_result_gradcam = stack_match_images(batch_images_gradcam, descriptions, vis_match_mask)

        results_cam_plusplus = self._gradcam(kwargs['query_idx'], kwargs['match_idx'], self.evaluator.model, plusplus=True)
        results_cam = self._gradcam(kwargs['query_idx'], kwargs['match_idx'], self.evaluator.model, plusplus=False)

        results_cam_onimage = self._gradcam_on_image(kwargs['query_idx'], kwargs['match_idx'], self.evaluator.model, plusplus=True)

        (query_image_heatmap_plusplus, match_image_heatmap_plusplus) = self.heatmap(results_cam_plusplus, kwargs['query_idx'], kwargs['match_idx'])
        (query_image_heatmap, match_image_heatmap) = self.heatmap(results_cam, kwargs['query_idx'], kwargs['match_idx'])
        (query_image_heatmap_onimage, match_image_heatmap_onimage) = self.heatmap(results_cam_onimage, kwargs['query_idx'], kwargs['match_idx'])

        (image_overlay_1, heatmap_point_1, image_overlay_2, heatmap_point_2) = self.point_specific_map(self.paths[kwargs['query_idx']], self.paths[kwargs['match_idx']])

        return {
            "figure": (vis_result_gradcam_plus_plus, vis_result_gradcam, image_overlay_1, image_overlay_2),
            "query_heatmaps": (query_image_heatmap_plusplus, query_image_heatmap, heatmap_point_1, query_image_heatmap_onimage),
            "match_heatmaps": (match_image_heatmap_plusplus, match_image_heatmap, heatmap_point_2, match_image_heatmap_onimage)
        }
    
if __name__ == "__main__":
     root = Path("/srv/transparency/wildbook_prototype/data/beluga_example_miewid")
     checkpoint_path = Path("/srv/transparency/wildbook_prototype/data/beluga-model-data/beluga-single-species.bin")