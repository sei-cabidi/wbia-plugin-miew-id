from torch.utils.data import Sampler

class IdxSampler(Sampler):
    """Samples elements sequentially, in the order of indices"""
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
