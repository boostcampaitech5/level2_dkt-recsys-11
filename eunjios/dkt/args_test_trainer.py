import math
import os
import gc
import copy

import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid
import wandb
from sklearn.model_selection import KFold

from .criterion import get_criterion
# from .dataloader import get_loaders
from .args_test_dataloader import get_loaders, get_loaders_kfold
from .metric import get_metric
from .args_test_model import LastQuery 
from .optimizer import get_optimizer
from .args_test_scheduler import get_scheduler
from .utils import get_logger, logging_conf

# ========= ADD: 파일명 지정 =========
from datetime import datetime, timezone, timedelta
# =================================

logger = get_logger(logger_conf=logging_conf)


def run(args,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        model: nn.Module):

    train_loader, valid_loader = get_loaders(args=args, train=train_data, valid=valid_data)

    # For warmup scheduler which uses step interval
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        logger.info("Start Training: Epoch %s", epoch + 1)

        # TRAIN
        train_auc, train_acc, train_loss = train(train_loader=train_loader,
                                                 model=model, optimizer=optimizer,
                                                 scheduler=scheduler, args=args)

        # VALID
        auc, acc = validate(valid_loader=valid_loader, model=model, args=args)

        wandb.log(dict(epoch=epoch,
                       train_loss_epoch=train_loss,
                       train_auc_epoch=train_auc,
                       train_acc_epoch=train_acc,
                       valid_auc_epoch=auc,
                       valid_acc_epoch=acc))
        
        if auc > best_auc:
            best_auc = auc
            # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(state={"epoch": epoch + 1,
                                   "state_dict": model_to_save.state_dict()},
                            model_dir=args.model_dir,
                            model_filename="best_model.pt")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter, args.patience
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)
         # =========== ADD ===============
        elif args.scheduler == "cosine_annealing":
            scheduler.step()
        # ================================


def run_kfold(args, 
              train_data: np.ndarray, 
              preprocess, 
              model: nn.Module):
    # 사용하지 않는 메모리 비우기 
    torch.cuda.empty_cache()
    gc.collect()


    # ========= ADD: kfold using SubsetRandomSampler =========
    kfold = KFold(n_splits=args.kfold, random_state=args.seed, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_data)):
        # k fold 진행 
        inner_model = copy.deepcopy(model)

        # reset wandb for every fold
        wandb.init(entity="new-recs", project="dkt-kfold", config=vars(args))
        wandb.run.name = f"{args.model}{args.memo}_fold:{fold}{args.memo}"
        logger.info(f'====== Fold: {fold} ======')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) 
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)


        # ====== ADD: kfold train_loader, valid_loader =======
        # train loader
        train_loader = get_loaders_kfold(
            args,
            train_data,
            train_subsampler,
        )
        # valid loader 
        valid_loader = get_loaders_kfold(
            args,
            train_data,
            valid_subsampler,
        )
        # ======================================================

        # For warmup scheduler which uses step interval
        args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
            args.n_epochs
        )
        args.warmup_steps = args.total_steps // 10

        optimizer = get_optimizer(inner_model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        early_stopping_counter = 0
        for epoch in range(args.n_epochs):

            logger.info("Start Training: Epoch %s", epoch + 1)

            ### TRAIN
            train_auc, train_acc, train_loss = train(train_loader=train_loader, 
                                                     model=inner_model, 
                                                     optimizer=optimizer, 
                                                     scheduler=scheduler, 
                                                     args=args)

            ### VALID
            auc, acc = validate(valid_loader=valid_loader, model=inner_model, args=args)

            wandb.log(dict(epoch=epoch,
                       train_loss_epoch=train_loss,
                       train_auc_epoch=train_auc,
                       train_acc_epoch=train_acc,
                       valid_auc_epoch=auc,
                       valid_acc_epoch=acc))
            
            if auc > best_auc:
                best_auc = auc
                # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = inner_model.module if hasattr(inner_model, "module") else inner_model
                save_checkpoint(state={"epoch": epoch + 1,
                                       "state_dict": model_to_save.state_dict()},
                                model_dir=args.model_dir,
                                model_filename=f"{args.model}_fold_{fold}.pt")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter, args.patience
                    )
                    break

            # scheduler
            if args.scheduler == "plateau":
                scheduler.step(best_auc)
            #  # =========== ADD ===============
            # elif args.scheduler == "cosine_annealing":
            #     scheduler.step()
            # # ================================

        # finish wandb for every fold
        wandb.finish()

def train(train_loader: torch.utils.data.DataLoader,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
 
    for step, batch in enumerate(train_loader):
        # ============ ADD: dictionary -> array 형태로 변환 ========
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(list(batch.values()))
        targets = batch['answerCode']
        # ========================================================
        loss = compute_loss(preds=preds, targets=targets)
        update_params(loss=loss, model=model, optimizer=optimizer,
                      scheduler=scheduler, args=args)

        if step % args.log_steps == 0:
            logger.info("Training steps: %s Loss: %.4f", step, loss.item())

        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc, loss_avg


def validate(valid_loader: nn.Module, model: nn.Module, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        # ============ ADD: dictionary -> array 형태로 변환 ========
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(list(batch.values())) # 수정 
        targets = batch["answerCode"]
        # ========================================================

        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(args, test_data: np.ndarray, model: nn.Module) -> None:
    model.eval()
    _, test_loader = get_loaders(args=args, train=None, valid=test_data)
    total_preds = []
    for step, batch in enumerate(test_loader):
        # ============ ADD: dictionary -> array 형태로 변환 ========
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(list(batch.values())) # 수정 
        # ========================================================
        # predictions
        preds = sigmoid(preds[:, -1])
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)
    
    #### 파일 생성 ####
    KST = timezone(timedelta(hours=9))
    record_time = datetime.now(KST)
    csv_file_name = f"{args.model}_submission_{record_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    write_path = os.path.join(args.output_dir, csv_file_name)
    ####
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
            
    ### csv 기록 추가 ###
    log_write_path = os.path.join(args.output_dir, "prediction.log")
    args_list = [
                    "seed",
                    "max_seq_len",
                    "hidden_dim",
                    "n_layers",
                    "n_heads",
                    "dim_div",
                    "n_epochs",
                    "batch_size",
                    "lr",
                    "clip_grad",
                    "patience",
                    "log_steps",
                    "optimizer",
                    "scheduler"
                ]
    
    with open(log_write_path, "a", encoding="utf8") as w:
        w.write(f"{csv_file_name}\n")
        for arg_name in args_list:
            arg_value = getattr(args, arg_name)
            w.write(f"{arg_name}: {arg_value}\n")
        w.write("\n")
    ###
        
    logger.info("Successfully saved submission as %s", write_path)


# =========== 기존 inference 에서 fold 만 추가하여 파일명 설정 ===========
def inference_kfold(args, 
                    test_data: np.ndarray, 
                    model: nn.Module, 
                    fold: int) -> None: # fold 추가 
    model.eval()
    _, test_loader = get_loaders(args=args, train=None, valid=test_data)

    total_preds = []
    for step, batch in enumerate(test_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(list(batch.values())) # 수정 

        # predictions
        preds = sigmoid(preds[:, -1])
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    # 모델명_fold정보_submission_시간정보.csv
    KST = timezone(timedelta(hours=9))
    record_time = datetime.now(KST)
    write_path = os.path.join(args.output_dir, f"{args.model}_{fold}_submission_{record_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)
# ==================================================================


def get_model(args) -> nn.Module:

    try:
        model_name = args.model.lower()
        model = {
            "lastquery": LastQuery,
        }.get(model_name)(args)

    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name, args)
        raise e
    return model


def compute_loss(preds: torch.Tensor, targets: torch.Tensor):
    """
    loss계산하고 parameter update
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(pred=preds, target=targets.float())

    # 마지막 시퀀스에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss: torch.Tensor,
                  model: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  args):
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """ Saves checkpoint to a given directory. """
    save_path = os.path.join(model_dir, model_filename)
    logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)


def load_model(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model


# ============== 기존 load_model과는 다르게 .pt 파일로 불러옴 ==================
def load_model_kfold(args, fold: int):
    # fold 번째 pt를 불러옴 
    model_path = os.path.join(args.model_dir, f"{args.model}_fold_{fold}.pt")
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model
# ========================================================================
