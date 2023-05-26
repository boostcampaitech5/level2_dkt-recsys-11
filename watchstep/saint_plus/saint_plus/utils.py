import os
import random
import yaml
import numpy as np
import pandas as pd
import torch

# config 가져오기
def get_config(config_path="/opt/ml/input/code/saint_plus/saint_plus/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.full_load(f)
    return config
    
# set random seed
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
        
def get_submission(preds, submit_version, output_dir='../outputs',):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_path = os.path.join(output_dir, f'submission_{submit_version}.csv')
    submission = pd.read_csv('../data/sample_submission.csv')
    submission['prediction'] = preds
    
    submission.to_csv(save_path, index=False)