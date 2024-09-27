import pickle
import torch

from tqdm import tqdm
from pathlib import Path

def unpickler(fp: Path | str):
    if isinstance(fp, str):
        fp = Path(fp)

    try:
        with open(fp, "r") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(fp, "rb") as f:
            return pickle.load(f)

def load_miewid_data(match_path: str | Path):
    ### df_test stores the dataframe containing metadata for the N instances of the test dataset
    df_test = unpickler(Path(match_path, "df_test.pkl"))

    ### test_dataset stores the actual MiewIDDataset object used during evaluation
    test_dataset = unpickler(Path(match_path, "test_dataset.pkl"))

    ### match_results stores 4 tensors:
    ###     match_results[0] := (N)-dimensional array of true PID
    ###     match_results[1] := (N,k)-dimensional tensor of top-k index guesses. Each row is the top-k indices *in the match matrix* for a single instance.
    ###     match_results[2] := (N,k)-dimensional tensor of top-k true PIDs. Each row is the top-k guesses for a single instance.
    ###     match_results[3] := (N,k)-dimensional tensor of the match matrix indicating success/failure of the top-k guesses
    match_results = unpickler(Path(match_path, "match_results.pkl"))
    q_pids = unpickler(Path(match_path, "q_pids.pkl")) # identical to match_results[0]
    topk_idx = unpickler(Path(match_path, "topk_idx.pkl")) # identical to match_results[1]
    topk_names = unpickler(Path(match_path, "topk_names.pkl")) # identical to match_results[2]
    match_mat = unpickler(Path(match_path, "match_mat.pkl")) # identical to match_results[3]

    ### distmat stores the (N,N)-dimensional pairwise distance matrix between the embeddings for all N instances in the test set
    distmat = unpickler(Path(match_path, "distmat.pkl"))

    # Iterate throught test dataset
    data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
    )
    batched_images = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for batch in tk0:
            batched_images.append(batch["image"])

    # Combine images into a single tensor
    images = torch.stack(batched_images).squeeze(dim=1)
    print(f"Loaded images: {images.shape}")
    del batched_images

    return df_test, test_dataset, match_results, q_pids, topk_idx, topk_names, match_mat, distmat, images