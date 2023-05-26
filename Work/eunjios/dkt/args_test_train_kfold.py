import os

import numpy as np
import torch
import wandb

from dkt import args_test_trainer_kfold
from dkt.args import parse_args
from dkt.args_test_dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)

# TODO: args_test_train.py 와 파일 합치기 
def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name) # group.values
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)
    
    logger.info("Building Model ...")
    model: torch.nn.Module = args_test_trainer_kfold.get_model(args=args).to(args.device)
    

    logger.info("Start Training ...")
    args_test_trainer_kfold.run_kfold(args=args, train_data=train_data, preprocess=preprocess, model=model)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
