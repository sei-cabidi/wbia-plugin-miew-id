import torch
import numpy as np

def precision_at_k(names, distmat, names_db=None, ranks=list(range(1, 21)), return_matches=False):
    """Computes precision at k given a distance matrix. 
    Assumes the distance matrix is square and does one-vs-all evaluation"""
    # assert distmat.shape[0] == distmat.shape[1], "Distance matrix must be square"


    if names_db is None or np.array_equal(names, names_db):
        names_db = names
        square_offset = 1
    else:
        square_offset = 0

    output = torch.Tensor(distmat[:, :]) * -1
    y = torch.Tensor(names[:]).squeeze(0)
    ids_tensor = torch.Tensor(names_db)

    max_k = max(ranks)

    topk_idx = output.topk(max_k+square_offset)[1][:, square_offset:] ### 
    topk_names = ids_tensor[topk_idx]

    match_mat = topk_names == y[:, None].expand(topk_names.shape)

    scores = []
    for k in ranks:
        match_mat_k = match_mat[:, :k]
        rank_mat = match_mat_k.any(axis=1)

        score = rank_mat.sum() / len(rank_mat)
        scores.append(score)

    if return_matches:
        return scores, match_mat, topk_idx, topk_names
    else:
        return scores




    

def topk_average_precision(names, distmat, names_db=None, k=None):
    """Computes top-k average precision given a distance matrix.
    Assumes the distance matrix is square and does one-vs-all evaluation"""
    # assert distmat.shape[0] == distmat.shape[1], "Distance matrix must be square"

    if names_db is None or np.array_equal(names, names_db):
        names_db = names
        square_offset = 1
    else:
        square_offset = 0

    output = torch.Tensor(distmat[:, :]) * -1
    y = torch.Tensor(names[:]).squeeze(0)
    ids_tensor = torch.Tensor(names_db)

    if k==None: 
        k = output.shape[1] - square_offset #### - 1
    score_array = torch.tensor([1.0 / i for i in range(1, k + 1)], device=output.device)

    _topk = output.topk(k+square_offset)[1][:, square_offset:] #### k+1 1:
    topk = ids_tensor[_topk]

    match_mat = topk == y[:, None].expand(topk.shape)
    rel_mat = match_mat.sum(axis=1)
    cum_mat = match_mat.cumsum(dim=1)

    ap_mat = ((cum_mat * score_array) * match_mat).sum(axis=1) / rel_mat

    return ap_mat.nan_to_num(0).mean().item()
