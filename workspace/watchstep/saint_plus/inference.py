import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from saint_plus.dataset import get_dataloaders
from saint_plus.model import SAINTPlus
from saint_plus.utils import get_config, seed_everything


def main():
  config = get_config()

  seed_everything(config.seed)    
  use_cuda: bool = config.use_cuda and torch.cuda.is_available() 
  device = torch.device("cuda" if use_cuda else "cpu")
  logger = WandbLogger(log_model=False)
  trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                       default_root_dir='../outputs/',
                       logger=logger)
  