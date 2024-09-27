import typing as tp
import pandas as pd
import numpy as np
import argparse
import os

from wbia_miew_id.helpers import get_config
from custom_utils import load_miewid_data
from pathlib import Path
from tqdm import tqdm

from GradCamPlusPlusVisualizer import GradCamPlusPlusVisualizer
from LightGlueVisualizer import LightGlueVisualizer
from visualizer import Visualizer

def pipeline(
        visualizer: Visualizer,
        images: tp.List[np.ndarray],
        topk_idx: np.ndarray,
        match_mat: np.ndarray,
        dist_mat: np.ndarray,
        ):
    # Iterate through instance-match-indicator tuples
    idx_to_match_idxs = {
        idx:{
            "topk_indices":     topk_idx[idx].tolist(),
            "match_indicators": match_mat[idx].tolist()
            }
        for idx in range(len(images))
    }
    results = []
    for query_idx, vals in tqdm(idx_to_match_idxs.items()):
        image_query = images[query_idx]

        # Extract top-k indices of matches and whether they were true matches
        topk_indices = vals["topk_indices"]
        match_indicators = vals["match_indicators"]

        # Iterate through top-k matches
        for rank, (match_idx, is_match) in enumerate(zip(topk_indices, match_indicators)):
            image_match = images[match_idx]
            
            print(f"Vizualizing query {query_idx} and match {match_idx}")

            # Generate results from the visualization method
            result = visualizer.generate(
                image_query,
                image_match,
                is_match=is_match,
                topk_indices=topk_indices
            )

            score = dist_mat[query_idx, match_idx]

            # Store computed results
            data = {
                "query_idx":    query_idx,
                "match_idx":    match_idx,
                "score":        score,
                "rank":         rank,
                "is_match":     is_match,
                **result
            }
            
            results.append(data)

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default="../wbia_miew_id/examples/beluga_example_miewid/benchmark_model/miew_id.msv2_all.yaml",
        help="Path to the YAML configuration file. Default: ../wbia_miew_id/examples/beluga_example_miewid/benchmark_model/miew_id.msv2_all.yaml",
    )

    parser.add_argument(
        "--root",
        type=str,
        default="../wbia_miew_id/examples",
        help="Root path where example data lives. Default: ../wbia_miew_id/examples",
    )
    
    parser.add_argument(
        "--savepath",
        type=str,
        default="/srv/transparency/wildbook_prototype/data/matches/figures/lightglue/",
        help="Path to store generated visualizations. Default: /srv/transparency/wildbook_prototype/data/matches/figures/lightglue/",
    )


    return parser.parse_args()

if __name__ == "__main__":
    #########################################################
    #                       Data Setup                      #
    #########################################################
    args = parse_args()
    config_path = args.config
    config = get_config(config_path)
    root = Path(args.root)
    visualization_output_dir = Path(root, f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}/visualizations")
    
    ### Load precomputed matches generated by MiewID
    df_test, test_dataset, match_results, q_pids, topk_idx, topk_names, match_mat, dist_mat, images = load_miewid_data(visualization_output_dir)

    ### Set up path to save images
    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)

    # Set up dataframe to store computed results
    df_all = pd.DataFrame(columns=[])

    #########################################################
    #               Visualization Generation                #
    #########################################################
    lightglue_results = pipeline(
                            LightGlueVisualizer(),
                            images,
                            topk_idx,
                            match_mat,
                            dist_mat
    )
    gradcam_results = pipeline(
                            GradCamPlusPlusVisualizer(),
                            images,
                            topk_idx,
                            match_mat,
                            dist_mat
    )

    #########################################################
    #               Map to common viz space                 #
    #########################################################
    # TODO: Grab code from common-visualization-space.ipynb