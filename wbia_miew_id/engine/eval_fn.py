import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import wandb

from metrics import AverageMeter, compute_distance_matrix, eval_onevsall, topk_average_precision, precision_at_k
from torch.cuda.amp import autocast  

def extract_embeddings(data_loader, model, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tk0:
            with autocast():
                batch_embeddings = model.extract_feat(batch["image"].to(device))
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            image_idx = batch["image_idx"].tolist()
            batch_embeddings_df = pd.DataFrame(batch_embeddings, index=image_idx)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch['label'].tolist()
            labels.extend(batch_labels)
            
    embeddings = pd.concat(embeddings)
    embeddings = embeddings.values

    assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"

    return embeddings, labels

def calculate_matches(embeddings, labels, embeddings_db=None, labels_db=None, dist_metric='cosine', ranks=list(range(1, 21))):

    q_pids = np.array(labels)
    
    qf = torch.Tensor(embeddings)
    if embeddings_db is not None:
        print('embeddings_db not note')
        df = torch.Tensor(embeddings_db)
        labels_db = np.array(labels_db)
    else:
        dbf = qf
        
    distmat = compute_distance_matrix(qf, dbf, dist_metric)

    distmat = distmat.numpy()

    print("Computing CMC and mAP ...")

    mAP = topk_average_precision(q_pids, distmat, labels_db, k=None)
    cmc, match_mat, topk_idx, topk_names = precision_at_k(q_pids, distmat, labels_db, ranks=ranks, return_matches=True)
    print(f"Computed rank metrics on {match_mat.shape[0]} examples")

    return mAP, cmc, (embeddings, q_pids, distmat)

def log_results(mAP, cmc, use_wandb=True):
    ranks=[1, 5, 10, 20]
    print("** Results **")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        if use_wandb: wandb.log({"Rank-{:<3}".format(r): cmc[r - 1]})
        
    if use_wandb: wandb.log({"mAP": mAP})

def eval_fn(data_loader, model, device, use_wandb=True, return_outputs=False):

    embeddings, labels = extract_embeddings(data_loader, model, device)

    mAP, cmc, (embeddings, q_pids, distmat) = calculate_matches(embeddings, labels)

    log_results(mAP, cmc, use_wandb=use_wandb)

    if return_outputs:
        return mAP, cmc, (embeddings, q_pids, distmat)
    else:
        return mAP, cmc