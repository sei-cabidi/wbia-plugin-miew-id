import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from .train_fn import train_fn
from .eval_fn import eval_fn

@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    This is a copy of torch.optim.swa_utils.update_bn that allows us to
    pass in a DataLoader that has batches of dictionaries with the image
    stored in a field called 'input'

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        elif isinstance(input, dict):
            label = input['label']
            input = input['image']
        if device is not None:
            input = input.to(device)
            label = label.to(device)

        model(input, label=label)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

def run_fn(config, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):

    best_score = 0
    for epoch in range(config.engine.epochs):

        train_loss = train_fn(train_loader, model,criterion, optimizer, device,scheduler=scheduler,epoch=epoch, use_wandb=use_wandb, swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler)

        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.bin')
        
        valid_score, cmc, output = eval_fn(valid_loader, model, device, use_wandb=use_wandb, return_outputs=True)
        diagram = output[-1]
        diagram.savefig(f"reliability_diagram_val_{epoch}.pdf")
            
        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), f'{checkpoint_dir}/model_best.bin')
            print('best model found for epoch {}'.format(epoch))

    if swa_model:
        # Update bn statistics for the swa_model at the end
        print("Updating SWA batchnorm statistics...")
        update_bn(train_loader, model, device=device)
        torch.save(swa_model.state_dict(), f'{checkpoint_dir}/swa_model_{epoch}.bin')
        #valid_score_swa, cmc_swa = eval_fn(valid_loader, swa_model, device, use_wandb=use_wandb)
    
    return best_score