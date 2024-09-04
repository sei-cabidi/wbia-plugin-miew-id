import numpy as np
import torch
import abc

class Visualizer(abc.ABC):
    pass
    
    def generate(self, image_query: np.ndarray | torch.Tensor, image_match: np.ndarray | torch.Tensor) -> ...:
        raise NotImplementedError