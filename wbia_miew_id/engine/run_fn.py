import torch
from tabulate import tabulate

from .train_fn import train_fn
from .eval_fn import eval_fn
from helpers.swatools import update_bn
def run_fn(config, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):

    best_score = 0
    for epoch in range(config.engine.epochs):
        train_loss = train_fn(train_loader, model,criterion, optimizer, device,scheduler=scheduler,epoch=epoch, use_wandb=use_wandb, swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler)

        # print("\nGetting metrics on train set...")
        # train_score, train_cmc = eval_fn(train_loader, model, device, use_wandb=use_wandb, return_outputs=False)
        
        print("\nGetting metrics on validation set...")
        valid_score, valid_cmc = eval_fn(valid_loader, model, device, use_wandb=use_wandb, return_outputs=False)

        # print("\n")
        # print(tabulate([["Train", 0], ["Valid", valid_score]], headers=["Split", "mAP"]))
        # print("\n\n")

        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), f'{checkpoint_dir}/model_best.bin')
            print('best model found for epoch {}'.format(epoch))

    # Update bn statistics for the swa_model at the end
    if swa_model:
        print("Updating SWA batchnorm statistics...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), f'{checkpoint_dir}/swa_model_{epoch}.bin')
    
    return best_score