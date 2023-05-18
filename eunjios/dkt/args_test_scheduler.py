import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
    # =========== ADD:cosine_annealing ==================
    elif args.scheduler == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    # ===================================================
    return scheduler
