import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from saint_plus.dataset import get_dataloaders
from saint_plus.model import SAINTPlus
from saint_plus.utils import get_config, seed_everything


def main():
  wandb.login()
  config = get_config()
  wandb.init(entity="new-recs", project="dkt", config=config)
    
  seed_everything(config.seed)    
  use_cuda: bool = config['no_cuda'] and torch.cuda.is_available() 
  device = torch.device("cuda" if use_cuda else "cpu")

  train_loader, val_loader = get_dataloaders()
    
  saint_plus = SAINTPlus()
  saint_plus = saint_plus.to(device)
  wandb_logger = WandbLogger(log_model=False)
  trainer = pl.Trainer(gpus=torch.cuda.device_count(), max_epochs=10, progress_bar_refresh_rate=21, logger=wandb_logger)
  
  trainer.fit(model=saint_plus,
              train_dataloader=train_loader,
              val_dataloaders=[val_loader, ],
              callbacks=[
                EarlyStopping(
                  monitor="avg_val_auc", 
                  patience=5, 
                  mode="max"
                ),
                ModelCheckpoint(
                  monitor="avg_val_auc",
                  filename="saint_plus-{epoch}-{val_loss_step:.2f}-{avg_val_auc:.2f}",
                  mode="max",
                ),
              ])
  
if __name__ == "__main__":
  main()
    
    