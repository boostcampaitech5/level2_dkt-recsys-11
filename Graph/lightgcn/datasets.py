import os
from typing import Tuple

import pandas as pd
import torch

from lightgcn.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


### item 지정 추가 ###
def prepare_dataset(device: str, data_dir: str, item: str) -> Tuple[dict, dict, int]:
    data = load_data(data_dir=data_dir, item=item)
    train_data, test_data = separate_data(data=data)
    id2index: dict = indexing_data(data=data, item=item)
    train_data_proc = process_data(data=train_data, id2index=id2index, device=device, item=item)
    test_data_proc = process_data(data=test_data, id2index=id2index, device=device, item=item)

    print_data_stat(train_data, "Train", item)
    print_data_stat(test_data, "Test", item)

    return train_data_proc, test_data_proc, len(id2index)


def load_data(data_dir: str, item: str) -> pd.DataFrame: 
    
    path1 = os.path.join(data_dir, "train_data.csv")
    path2 = os.path.join(data_dir, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data = pd.concat([data1, data2])

    ### 중복은 마지막 데이터만 남기기 ###
    data.drop_duplicates(subset=["userID", item], keep="last", inplace=True)
    
    return data


def separate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]
    return train_data, test_data


def indexing_data(data: pd.DataFrame, item: str) -> dict:
    userid, itemid = (
        sorted(list(set(data.userID))),
        ### item 지정 추가 ###
        sorted(list(set(data[item]))),
    )
    n_user, n_item = len(userid), len(itemid)
    userid2index = {v: i for i, v in enumerate(userid)}
    itemid2index = {str(v): i + n_user for i, v in enumerate(itemid)}
    id2index = dict(userid2index, **itemid2index)
    return id2index


def process_data(data: pd.DataFrame, id2index: dict, device: str, item: str) -> dict:
    edge, label = [], []
    for user, item, acode in zip(data.userID, data[item], data.answerCode):
        ### item 지정 추가 ###
        uid, iid = id2index[user], id2index[str(item)]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    return dict(edge=edge.to(device),
                label=label.to(device))


def print_data_stat(data: pd.DataFrame, name: str, item: str) -> None:
    ### item 지정 추가 ###
    userid, itemid = list(set(data.userID)), list(set(data[item]))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
