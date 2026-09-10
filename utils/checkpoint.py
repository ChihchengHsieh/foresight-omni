import torch, os
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
from datetime import datetime


def save_dist(
    args,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
):

    if args.distributed:
        if dist.get_rank() == 0:
            saved_path = save_checkpoint(
                output_dir=args.output_dir,
                model=model.module,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            print(f"Model is saved to {saved_path}")
            return saved_path
    else:
        saved_path = save_checkpoint(
            output_dir=args.output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"Model is saved to {saved_path}")
        return saved_path


def save_checkpoint(
    output_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    epoch: Optional[int] = None,
):
    saving_dict = {"model": model.state_dict()}
    if optimizer:
        saving_dict["optimizer"] = optimizer.state_dict()

    if scheduler:
        saving_dict["scheduler"] = scheduler.state_dict()

    if epoch is not None:
        saving_dict["epoch"] = epoch

    # os.makedirs("checkpoints", exist_ok=True)
    # os.makedirs(saving_folder, exist_ok=True)
    torch.save(
        saving_dict,
        os.path.join(output_dir, "model"),
    )
    return output_dir


def load_checkpoint(model_name: str, device):
    cp = torch.load(
        os.path.join("checkpoints", model_name, "model"), map_location=device
    )
    return cp


def get_clear_time_str(d: datetime):
    return f"{d.year}_{d.month}_{d.day}_{d.hour}_{d.minute}_{d.second}"


def save_checkpoint_with_affix(
    args: object,
    output_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    epoch: Optional[int] = None,
    affix: str = "",
):
    saving_dict = {"model": model.state_dict()}
    if optimizer:
        saving_dict["optimizer"] = optimizer.state_dict()

    if scheduler:
        saving_dict["scheduler"] = scheduler.state_dict()

    if epoch is not None:
        saving_dict["epoch"] = epoch

    # also save args.
    saving_dict["args"] = args

    # os.makedirs("checkpoints", exist_ok=True)
    # os.makedirs(saving_folder, exist_ok=True)

    model_saving_path = os.path.join(output_dir, f"{affix}model")
    torch.save(
        saving_dict,
        model_saving_path,
    )

    return model_saving_path


def load_checkpoint_from_path(path: str, device):
    cp = torch.load(path, map_location=device)
    return cp


def load_continue_training(
    path: str,
    device,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
):
    cp = load_checkpoint_from_path(path, device)
    model.load_state_dict(cp["model"], strict=True)

    if optimizer:
        optimizer.load_state_dict(cp["optimizer"])

    if scheduler:
        scheduler.load_state_dict(cp["scheduler"])

    epoch = cp.get("epoch", get_trained_epoch_from_name(os.path.basename(path)))
    print(f"Model is loaded from [{path}]")

    del cp
    torch.cuda.empty_cache()
    return model, optimizer, scheduler, epoch


def get_trained_epoch_from_name(name: str):
    return int(name.split("_")[1])


def save_dist_with_time(
    args,
    model: nn.Module,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    affix:str = "",
):
    time_str = get_clear_time_str(datetime.now())

    if args.distributed:
        if dist.get_rank() == 0:
            saved_path = save_checkpoint_with_affix(
                args=args,
                output_dir=args.output_dir,
                model=model.module,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                affix=f"epoch_{epoch}_{time_str}_{affix}",
            )
            print(f"Model is saved to {saved_path}")
            return saved_path
    else:
        saved_path = save_checkpoint_with_affix(
            args=args,
            output_dir=args.output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            affix=f"epoch_{epoch}_{time_str}_{affix}",
        )
        print(f"Model is saved to {saved_path}")
        return saved_path
