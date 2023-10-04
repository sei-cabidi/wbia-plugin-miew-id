import torch
from datasets import MiewIdDataset, get_test_transforms
from .eval_fn import eval_fn
from etl import filter_min_names_df, subsample_max_df

def group_eval(config, df_test, eval_groups, model):

    print("** Calculating groupwise evaluation scores **")

    group_results = []
    for eval_group in eval_groups:
        for group, df_group in df_test.groupby(eval_group):
            print('* Evaluating group:', group)
            n_filter_min = config.data.test.n_filter_min
            if n_filter_min:
                print(len(df_group))
                df_group = filter_min_names_df(df_group, n_filter_min)
            n_subsample_max = config.data.test.n_subsample_max
            if n_subsample_max:
                df_group = subsample_max_df(df_group, n_subsample_max)
            test_dataset = MiewIdDataset(
                csv=df_group,
                transforms=get_test_transforms(config),
                fliplr=config.test.fliplr,
                fliplr_view=config.test.fliplr_view,
                crop_bbox=config.data.crop_bbox,
            )
                
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config.engine.valid_batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )
            try:
                device = torch.device(config.engine.device)
                test_score, test_cmc, test_outputs = eval_fn(test_loader, model, device, use_wandb=False, return_outputs=True)
            except Exception as E:
                print(E)
                test_score, test_cmc = 0, 0

            group_result = (group, test_score, test_cmc)

            group_results.append(group_result)

    return group_results