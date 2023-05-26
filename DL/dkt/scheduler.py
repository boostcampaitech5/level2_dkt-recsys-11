import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer: torch.optim.Optimizer, args):
    # Plateau
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    # Linear Warmup
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
    # Cosine Annealing
    elif args.scheduler == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    # Step LR
    elif args.scheduler == "steplr":
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)

    return scheduler
