import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
from metrics import AverageMeter, compute_calibration
from visualization.reliability_diagrams.reliability_diagrams import reliability_diagram


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):
    model.train()
    loss_score = AverageMeter()
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    all_targets = []
    all_images = []
    all_outputs = []
    for bi, d in tk0:
        images = d['image']
        targets = d['label']
        batch_size = images.shape[0]

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(images, targets)
        
        if epoch > 13:
            import pdb
            pdb.set_trace()
            # Are targets here the same as targets in eval_fn?

        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        batch_accuracy = (targets == torch.softmax(
            output, dim=1).argmax(dim=1)).sum() / targets.shape[0]

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch,
                        LR=optimizer.param_groups[0]['lr'], TrainAcc=batch_accuracy.item())

        all_outputs.append(output.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_images.append(images.detach().cpu())


    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets).numpy()
    all_images = torch.cat(all_images).numpy()
    confidences = torch.softmax(all_outputs, dim=1)
    pred_targets = confidences.argmax(dim=1).numpy()
    confidences = confidences.numpy().max(axis=1)
    
    if epoch > 28:
        import pdb
        pdb.set_trace()

    if swa_model and epoch > swa_start:
        print("Updating swa model...")
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    # Calibration metrics
    print("Computing ECE ...")
    metrics = compute_calibration(
        all_targets, pred_targets, confidences, num_bins=10)
    ECE = metrics['expected_calibration_error']
    print(f"Epoch {epoch} ECE: {ECE}")

    print("Drawing reliability diagram...")
    diagram = reliability_diagram(
        all_targets, pred_targets,  confidences, num_bins=10, return_fig=True)
    diagram.savefig(f"reliability_diagram_train_{epoch}.pdf")
    plt.close()

    if use_wandb:
        wandb.log({
            "train loss": loss_score.avg,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr'],
            "batch_accuracy": batch_accuracy
        })

    return loss_score
