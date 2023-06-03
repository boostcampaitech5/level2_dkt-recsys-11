import wandb
import yaml
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from saint_plus.utils import get_config
from saint_plus.model import EncoderEmbedding, DecoderEmbedding, StackedNMultiHeadAttention

config = get_config()

class SAINTPlusModule(nn.Module):
  def __init__(self):
    # n_encoder,n_detotal_responses,seq_len,max_time=300+1
    super(SAINTPlusModule, self).__init__()
    self.loss = nn.BCEWithLogitsLoss()
    self.encoder_layer = StackedNMultiHeadAttention(n_stacks=config.num_decoder,
                                                    n_dims=config.embed_dims,
                                                    n_heads=config.dec_heads,
                                                    seq_len=config.max_seq,
                                                    n_multihead=1, dropout=0.0)
    self.decoder_layer = StackedNMultiHeadAttention(n_stacks=config.num_encoder,
                                                    n_dims=config.embed_dims,
                                                    n_heads=config.enc_heads,
                                                    seq_len=config.max_seq,
                                                    n_multihead=2, dropout=0.0)
    self.encoder_embedding = EncoderEmbedding(n_exercises=config.total_exe,
                                              n_categories=config.total_cat,
                                              n_dims=config.embed_dims, 
                                              seq_len=config.max_seq)
    self.decoder_embedding = DecoderEmbedding(n_responses=3, 
                                              n_dims=config.embed_dims, 
                                              seq_len=config.max_seq)
    self.elapsed_time = nn.Linear(1, config.embed_dims)
    self.fc = nn.Linear(config.embed_dims, 1)

  def forward(self, x, y):
    enc = self.encoder_embedding(exercises=x["input_ids"],
                                 categories=x['input_cat'])
    dec = self.decoder_embedding(responses=y)
    elapsed_time = x["input_rtime"].unsqueeze(-1).float()
    ela_time = self.elapsed_time(elapsed_time)
    dec = dec + ela_time
    # encoder
    encoder_output = self.encoder_layer(input_k=enc,
                                        input_q=enc,
                                        input_v=enc)
    # decoder
    decoder_output = self.decoder_layer(input_k=dec,
                                        input_q=dec,
                                        input_v=dec,
                                        encoder_output=encoder_output,
                                        break_layer=1)
    # fully connected layer
    out = self.fc(decoder_output)
    return out.squeeze()

  def training_step(self, batch, batch_idx):
    input, labels = batch
    target_mask = (input["input_ids"] != 0)
    out = self(input, labels)
    loss = self.loss(out.float(), labels.float())
    out = torch.masked_select(out, target_mask)
    out = torch.sigmoid(out)
    labels = torch.masked_select(labels, target_mask)
    self.log("train_loss", loss, on_step=True, prog_bar=True)
    return {"loss": loss, "outs": out, "labels": labels}

  def training_epoch_end(self, training_ouput):
    out = np.concatenate([i["outs"].cpu().detach().numpy()
                              for i in training_ouput]).reshape(-1)
    labels = np.concatenate([i["labels"].cpu().detach().numpy()
                                 for i in training_ouput]).reshape(-1)
    auc = roc_auc_score(labels, out)
    acc = accuracy_score(labels, out)
    self.print("train_auc", auc)
    self.print("train_acc", acc)
    self.log("train_auc", auc)
    self.log("train_acc", acc)
    
  def validation_step(self, batch, batch_idx):
    input, labels = batch
    target_mask = (input["input_ids"] != 0)
    out = self(input, labels)
    loss = self.loss(out.float(), labels.float())
    out = torch.masked_select(out, target_mask)
    out = torch.sigmoid(out)
    labels = torch.masked_select(labels, target_mask)
    self.log("val_loss", loss, on_step=True, prog_bar=True)
    return {"val_loss": loss, "outs": out, "labels": labels}

  def validation_epoch_end(self, validation_ouput):
    out = np.concatenate([i["outs"].cpu().detach().numpy()
                              for i in validation_ouput]).reshape(-1)
    labels = np.concatenate([i["labels"].cpu().detach().numpy()
                                 for i in validation_ouput]).reshape(-1)
    auc = roc_auc_score(labels, out)
    acc = accuracy_score(labels, out)
    self.print("val auc", auc)
    self.print("val_acc", acc)
    self.log("val_auc", auc)
    self.log("val_acc", acc)
  
  # optimizer
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=config.lr)
  
  def test_step(self, batch, batch_idx):
    loss, auc = self._shared_eval_Step(batch, batch_idx)
    return loss, auc
  
  def predict_step(self, batch, batch_idx, dataloader_idx):
    input, labels = batch
    target_mask = (input["input_ids"] != 0)
    out = self(input, labels)
    loss = self.loss(out.float(), labels.float())
    out = torch.masked_select(out, target_mask)
    out = torch.sigmoid(out)
    labels = torch.masked_select(labels, target_mask)
    return labels
  
  


