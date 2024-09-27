import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import random
import torch
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import defaultdict
from torchvision import transforms
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from pathlib import Path
from tqdm import tqdm

import lightglue
from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import rbd

from .visualizer import Visualizer

class LightGlueVisualizer(Visualizer):
    def __init__(self):
        ### Set device as GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

        ### Load the SuperPoint extractor and put it on the GPU
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)

        ### Load the LightGlue matcher and put it on the GPU
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def plot_images(self, imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True, bordercolor='green'):
        """Plot a set of images horizontally.
        Args:
            imgs: list of NumPy RGB (H, W, 3) or PyTorch RGB (3, H, W) or mono (H, W).
            titles: a list of strings, as titles for each image.
            cmaps: colormaps for monochrome images.
            adaptive: whether the figure size should fit the image aspect ratios.
        """
        # conversion to (H, W, 3) for torch.Tensor
        imgs = [
            img.permute(1, 2, 0).cpu().numpy()
            if (isinstance(img, torch.Tensor) and img.dim() == 3)
            else img
            for img in imgs
        ]

        n = len(imgs)
        if not isinstance(cmaps, (list, tuple)):
            cmaps = [cmaps] * n

        if adaptive:
            ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
        else:
            ratios = [4 / 3] * n
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
            1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
        )
        if n == 1:
            ax = [ax]
        for i in range(n):
            ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
            ax[i].get_yaxis().set_ticks([])
            ax[i].get_xaxis().set_ticks([])
            #ax[i].set_axis_off()
            for spine in ax[i].spines.values():  # remove frame
                spine.set_visible(True)
                spine.set_edgecolor(bordercolor)
                spine.set_linewidth(8)
            if titles:
                ax[i].set_title(titles[i])

        fig.tight_layout(pad=pad)

        return fig, ax

    def plot_keypoints(self, kpts, colors="lime", ps=4, axes=None, a=1.0):
        """Plot keypoints for existing images.
        Args:
            kpts: list of ndarrays of size (N, 2).
            colors: string, or list of list of tuples (one for each keypoints).
            ps: size of the keypoints as float.
        """
        if not isinstance(colors, list):
            colors = [colors] * len(kpts)
        if not isinstance(a, list):
            a = [a] * len(kpts)
        if axes is None:
            axes = plt.gcf().axes
        for ax, k, c, alpha in zip(axes, kpts, colors, a):
            if isinstance(k, torch.Tensor):
                k = k.cpu().numpy()
            ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


    def plot_matches(self, kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
        """Plot matches for a pair of existing images.
        Args:
            kpts0, kpts1: corresponding keypoints of size (N, 2).
            color: color of each match, string or RGB tuple. Random if not given.
            lw: width of the lines.
            ps: size of the end points (no endpoint if ps=0)
            indices: indices of the images to draw the matches on.
            a: alpha opacity of the match lines.
        """
        fig = plt.gcf()
        if axes is None:
            ax = fig.axes
            ax0, ax1 = ax[0], ax[1]
        else:
            ax0, ax1 = axes
        if isinstance(kpts0, torch.Tensor):
            kpts0 = kpts0.cpu().numpy()
        if isinstance(kpts1, torch.Tensor):
            kpts1 = kpts1.cpu().numpy()
        assert len(kpts0) == len(kpts1)
        if color is None:
            color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
        elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
            color = [color] * len(kpts0)

        if lw > 0:
            for i in range(len(kpts0)):
                line = matplotlib.patches.ConnectionPatch(
                    xyA=(kpts0[i, 0], kpts0[i, 1]),
                    xyB=(kpts1[i, 0], kpts1[i, 1]),
                    coordsA=ax0.transData,
                    coordsB=ax1.transData,
                    axesA=ax0,
                    axesB=ax1,
                    zorder=1,
                    color=color[i],
                    linewidth=lw,
                    clip_on=True,
                    alpha=a,
                    label=None if labels is None else labels[i],
                    picker=5.0,
                )
                line.set_annotation_clip(True)
                fig.add_artist(line)

        # freeze the axes to prevent the transform to change
        ax0.autoscale(enable=False)
        ax1.autoscale(enable=False)

        if ps > 0:
            ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
            ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

        return fig, [ax0, ax1]

    def fig2np(self, fig) -> np.ndarray:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        return np.asarray(buf)

    def plot_stacked_figures(self, figs):
        image_queryrrays = [self.fig2np(fig) for fig in figs]
        f, axarr = plt.subplots(len(figs), 1, figsize=(16, 4 * len(figs)))
        for i in range(len(figs)):
            axarr[i].set_axis_off()
            axarr[i].get_yaxis().set_ticks([])
            axarr[i].get_xaxis().set_ticks([])
            axarr[i].imshow(image_queryrrays[i])
        plt.tight_layout()

    def plot_stacked_images(self, images):
        f, axarr = plt.subplots(len(images), 1, figsize=(16, 4 * len(images)))
        for i in range(len(images)):
            axarr[i].set_axis_off()
            axarr[i].get_yaxis().set_ticks([])
            axarr[i].get_xaxis().set_ticks([])
            axarr[i].imshow(images[i])
        plt.tight_layout()

    def plot_hulls(
            self,
            kpts0,
            kpts1,
            axes = None,
            kwargs: dict = {"dbscan_kwargs":{"eps":0.3, "min_samples":3}},
            return_output: bool = False
        ):
        # Figure and axes setup
        fig = plt.gcf()
        if axes is None:
            ax = fig.axes
            ax0, ax1 = ax[0], ax[1]
        else:
            ax0, ax1 = axes
        if isinstance(kpts0, torch.Tensor):
            kpts0 = kpts0.cpu().numpy()
        if isinstance(kpts1, torch.Tensor):
            kpts1 = kpts1.cpu().numpy()
        assert len(kpts0) == len(kpts1)
        
        # Compute the hulls for both sets of keypoints
        if len(kpts0) > 0 and len(kpts1) > 0:
            hulls_kpts0 = self.compute_convex_polygons(kpts0, dbscan_kwargs=kwargs["dbscan_kwargs"])
            hulls_kpts1 = self.compute_convex_polygons(kpts1, dbscan_kwargs=kwargs["dbscan_kwargs"])

            # Draw hull lines on left and right axes
            for hulls, ax in zip([hulls_kpts0, hulls_kpts1], [ax0, ax1]):
                for hull in hulls:
                    for simplex in hull.simplices:
                        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')

        if return_output:
            return fig, axes, hulls_kpts0, hulls_kpts1
            
        return fig, axes


    def compute_convex_polygons(
            self,
            X: ...,
            dbscan_kwargs: dict = {"eps":0.3, "min_samples":3}
        ) -> [ConvexHull]:
        """A polygon generator that uses the DBSCAN and ConvexHull algorithms from sklearn.

        Args:
            X (np.array, torch.Tensor): (num_samples,...)-shaped array of un-normalized data points
            dbscan_kwargs (dict, optional): kwargs to provide to DBSCAN. Defaults to {"eps":0.5, "min_samples":3}.

        Returns:
            output: a set of k  drawn in the same (un-normalized) coordinate system as X
        """
        if not isinstance(X, type(np.array)):
            X = np.array(X)
        
        # Normalize input data
        scaler = StandardScaler()
        scaler.fit(X)
        X_new = scaler.transform(X)
        # print(f"X mean: {scaler.mean_}\nX variance: {scaler.var_}")

        # Run DBSCAN
        # print(f"Running DBSCAN with args {dbscan_kwargs}")
        db = DBSCAN(**dbscan_kwargs).fit(X_new)

        # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        # n_noise_ = list(db.labels_).count(-1)
        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)

        # Extract the convex hull for each core sample set
        hulls = []
        for label in set(db.labels_):
            # Condition on samples matching label
            label_mask = np.zeros_like(db.labels_, dtype=bool)
            label_mask[db.labels_ == label] = True
            
            # Condition on samples in current core set
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            
            # Extract core sample set
            core_points = X[core_samples_mask & label_mask]

            # Compute convex hull if there are more samples than lower bound
            if len(core_points) > 3:
                hull = ConvexHull(core_points)
                hulls.append(hull)

        return hulls
        
        # Un-normalize data
        # hulls = np.array(hulls)
        # if len(hulls) > 0:
        #     hulls_denormalized = scaler.inverse_transform(hulls)
        #     return hulls_denormalized
        # return hulls

    def _lightglue_vis(
            self,
            image_query,
            image_match,
            feats0,
            feats1,
            matches01,
            is_match: bool,
            savepath: str | Path | None = None,
            polygon_kwargs: dict = {"dbscan_kwargs":{"eps":0.3, "min_samples":3}},
            return_output: bool = False
        ):
        def unnormalize(img_base):
            aug_mean = np.array([0.485, 0.456, 0.406])
            aug_std = np.array([0.229, 0.224, 0.225])
            unnormalize = transforms.Normalize((-aug_mean / aug_std).tolist(), (1.0 / aug_std).tolist())
            img_unnorm = unnormalize(img_base)

            return img_unnorm

        # Preprocessing
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()

        # Generate plots
        fig, axes = self.plot_images([unnormalize(image_query), unnormalize(image_match)], bordercolor='green' if is_match else 'red')
        fig, axes = self.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        #fig, axes, hulls0, hulls1 = self.plot_hulls(m_kpts0, m_kpts1, axes=axes, kwargs=polygon_kwargs, return_output=True) # Deprecated: polygons
        viz2d.add_text(0, f"num keypoints={len(m_kpts0)}", fs=20)

        # Save plots to disk
        if savepath:
            viz2d.save_plot(savepath)

        # Return computed data to form into a dataset
        if return_output:
            return fig, axes, (m_kpts0, m_kpts1)
            
        return fig, axes
    
    def generate(
            self,
            image_query: np.ndarray | torch.Tensor,
            image_match: np.ndarray | torch.Tensor,
            *,
            is_match: bool = False,
            **kwargs
        ) -> dict:
        """For a single query-match pair of images, generate a visualization and associated metadata."""
        # Extract and match features
        feats0 = self.extractor.extract(image_query.to(self.device))
        feats1 = self.extractor.extract(image_match.to(self.device))
        matches01 = self.matcher({"image0": feats0, "image1": feats1}) # Lightglue requires these to be called 'image0' and 'image1'

        # Visualize with lines
        fig, axs, (m_kpts0, m_kpts1)  = self._lightglue_vis(
            image_query,
            image_match,
            feats0,
            feats1,
            matches01,
            is_match=is_match,
            #polygon_kwargs={"dbscan_kwargs":{"eps":0.4, "min_samples":3}},
            return_output=True
        )

        # Render the figure
        image = self.fig2np(fig)

        # Store computed results
        data = {
            "figure":       image,
            "query_kpts":   [m_kpts0],
            "match_kpts":   [m_kpts1]
        }

        plt.close('all')
        
        return data