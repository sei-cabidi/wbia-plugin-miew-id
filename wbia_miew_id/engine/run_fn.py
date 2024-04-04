import torch
from tabulate import tabulate

from .train_fn import train_fn
from .eval_fn import eval_fn, group_eval_fn
from helpers.swatools import update_bn

def run_fn(config, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):

    best_score = 0

    #### To load the checkpoint
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # If you're using a scheduler
    # best_score = checkpoint['best_score']
    # start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch

    
    for epoch in range(config.engine.epochs):
        train_loss = train_fn(train_loader, model,criterion, optimizer, device,scheduler=scheduler,epoch=epoch, use_wandb=use_wandb, swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # If you're using a scheduler
            'best_score': best_score,
        }, f'{checkpoint_dir}/checkpoint_latest.bin')

        # print("\nGetting metrics on train set...")
        # train_score, train_cmc = eval_fn(train_loader, model, device, use_wandb=use_wandb, return_outputs=False)
        
        print("\nGetting metrics on validation set...")
        eval_groups = config.data.test.eval_groups
        
        if eval_groups:
            valid_score, valid_cmc = group_eval_fn(config, eval_groups, model)

        else:
            print('Evaluating on full test set')
            valid_score, valid_cmc = eval_fn(valid_loader, model, device, use_wandb=use_wandb, return_outputs=False)

        print('Valid score: ', valid_score)

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
