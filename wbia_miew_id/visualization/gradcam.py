import os
import time
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


from wbia_miew_id.datasets import MiewIdDataset, get_valid_transforms
from wbia_miew_id.models import MiewIdNet

def resize_image(image, new_height):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(new_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.6) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)

    # Keep heatmap areas lower than threshold transparent
    heatmap[mask <= 0.05] = 0
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img

    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def draw_one(config, test_loader, model, images_dir = '', method='gradcam_plus_plus', eigen_smooth=False, show=False):

    # Generate embeddings for query and db
    model.eval()
    tk0 = tqdm(test_loader, total=len(test_loader))
    embeddings = []
    labels = []
    images = []
    paths = []
    with torch.no_grad():   
        for batch in tk0:
            batch_image = batch[0]
            batch_name = batch[1]
            batch_path = batch[2]

            images.extend(batch_image)
            batch_embeddings = model(batch_image.to(config.engine.device))
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            batch_embeddings_df = pd.DataFrame(batch_embeddings)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch_name.tolist()
            labels.extend(batch_labels)
            
            paths = batch_path
            paths.extend(batch_path)
            
    embeddings = pd.concat(embeddings)

    target_layers = model.backbone.conv_head

    if method=='gradcam_plus_plus':
        generate_cam = GradCAMPlusPlus(model=model,target_layers=[target_layers],use_cuda=True)
    elif method=='eigencam':
        generate_cam = EigenCAM(model=model,target_layers=[target_layers],use_cuda=True)

    qry_idx = 0
    db_idx = 1

    qry_features = embeddings.iloc[qry_idx].values
    qry_features = torch.Tensor(qry_features).to(config.engine.device)

    db_features = embeddings.iloc[db_idx].values
    db_features = torch.Tensor(db_features).to(config.engine.device)

    similarity_to_qry = SimilarityToConceptTarget(qry_features)
    similarity_to_db = SimilarityToConceptTarget(db_features)

    qry_tensor = images[qry_idx]
    db_tensor = images[db_idx]

    db_tensor = db_tensor.unsqueeze(0)
    qry_tensor = qry_tensor.unsqueeze(0)

    qry_grayscale_cam = generate_cam(input_tensor=db_tensor,targets=[similarity_to_qry],aug_smooth=False,eigen_smooth=eigen_smooth)[0, :]
    db_grayscale_cam = generate_cam(input_tensor=qry_tensor,targets=[similarity_to_db],aug_smooth=False,eigen_smooth=eigen_smooth)[0, :]
    end = time.time()


    # query image results
    qry_image_path = paths[qry_idx]
    qry_float = load_image(qry_image_path)

    qry_float_norm = (qry_float - qry_float.min()) / (qry_float.max() - qry_float.min())
    db_grayscale_cam_res = cv2.resize(db_grayscale_cam, (qry_float_norm.shape[1], qry_float_norm.shape[0]))
    cam_image_qry = show_cam_on_image(qry_float_norm, db_grayscale_cam_res, use_rgb=True)

    ai0 = cam_image_qry
    ai1 = qry_float

    # db image results
    db_image_path = paths[db_idx]
    db_float = load_image(db_image_path)

    db_float_norm = (db_float - db_float.min()) / (db_float.max() - db_float.min())
    qry_grayscale_cam_res = cv2.resize(qry_grayscale_cam, (db_float_norm.shape[1], db_float_norm.shape[0]))
    cam_image_db = show_cam_on_image(db_float_norm, qry_grayscale_cam_res, use_rgb=True)

    ai2 = cam_image_db
    ai3 = db_float

    image_list = [ai0, ai1, ai2, ai3]
    resize_height = 440
    resized_image_list = [resize_image(img, resize_height) for img in image_list]
    comb_image = np.hstack(resized_image_list)
    if show:
        plt.imshow(comb_image)

    comb_image = cv2.cvtColor(comb_image, cv2.COLOR_BGR2RGB)
    return comb_image


