import torch

def precision_at_k(names, distmat, k=5, return_matches=False):
    """Computes precision at k given a distance matrix. 
    Assumes the distance matrix is square for one-vs-all evaluation"""
    assert distmat.shape[0] == distmat.shape[1], "Distance matrix must be square"

    output = torch.Tensor(distmat[:, :]) * -1
    y = torch.Tensor(names[:]).squeeze(0)
    ids_tensor = torch.Tensor(names)

    topk_idx = output.topk(k+1)[1][:, 1:]
    topk_names = ids_tensor[topk_idx]

    match_mat = topk_names == y[:, None].expand(topk_names.shape)
    rank_mat = match_mat.any(axis=1)

    score = rank_mat.sum() / len(rank_mat)

    if return_matches:
        return score, match_mat, topk_idx, topk_names
    else:
        return score