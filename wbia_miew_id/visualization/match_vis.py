import os
import cv2
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import Sampler


from .gradcam import draw_batch

class IdxSampler(Sampler):
    """Samples elements sequentially, in the order of indices"""
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def stack_match_images(images, descriptions, match_mask, text_color=(0, 0, 0)):  # OpenCV uses BGR
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

def render_single_query_result(config, model, vis_loader, df_vis, qry_row, qry_idx, vis_match_mask, k=5):
    
    
    use_cuda = False if config.engine.device in ['mps', 'cpu'] else True

    batch_images = draw_batch(
        config, vis_loader,  model, images_dir = 'dev_test', method='gradcam_plus_plus', eigen_smooth=False, 
        render_transformed=True, show=False, use_cuda=use_cuda)

    viewpoints = df_vis['viewpoint'].values
    names = df_vis['name'].values
    indices = df_vis.index.values
    qry_name = qry_row['name']
    qry_viewpoint = qry_row['viewpoint']

    desc_qry = [f"Query: {qry_name} {qry_viewpoint} ({qry_idx})" for i in range(len(viewpoints))]
    desc_db = [f"Match: {name} {viewpoint} ({idx})" for name, viewpoint, idx in zip(names, viewpoints, indices)]
    descriptions = [(q, d) for q, d in zip(desc_qry, desc_db)]

    vis_result = stack_match_images(batch_images, descriptions, vis_match_mask)

    output_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}/visualizations"
    output_name = f"vis_{qry_name}_{qry_viewpoint}_{qry_idx}_top{k}.jpg"
    output_path = os.path.join(output_dir, output_name)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, vis_result, [cv2.IMWRITE_JPEG_QUALITY, 60])

    print(f"Saved visualization to {output_path}")

def render_query_results(config, model, test_dataset, df_test, match_results, k=5):

    q_pids, topk_idx, topk_names, match_mat = match_results

    print("Generating visualizations...")
    for i in tqdm(range(len(q_pids))):
        #
        vis_idx = topk_idx[i].tolist()
        vis_idx

        vis_names = topk_names[i].tolist()
        vis_match_mask = match_mat[i].tolist()

        df_vis = df_test.iloc[vis_idx]
        qry_row = df_test.iloc[i]

        qry_idx = i
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
        
        render_single_query_result(config, model, vis_loader, df_vis, qry_row, qry_idx, vis_match_mask, k=k)
