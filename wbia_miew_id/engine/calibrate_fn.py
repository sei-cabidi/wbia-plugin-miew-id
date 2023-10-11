import wandb
import torch


def calibrate_fn(dataloader,model,device, checkpoint_dir,use_wandb=True):
    model.eval()    

    model.set_temperature(dataloader, device)

    if use_wandb:
        wandb.log({
            "new temperature": model.temperature,
            "post-TS NLL": model.after_temperature_nll,
            "post-TS ECE": model.after_temperature_ece
            })

    torch.save(model.state_dict(), f'{checkpoint_dir}/model_best_calibrated.bin')

    return model.after_temperature_ece

