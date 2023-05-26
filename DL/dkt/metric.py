from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def get_metric(targets: np.ndarray, preds: np.ndarray) -> Tuple[float]:
    auc = roc_auc_score(y_true=targets, y_score=preds)
    acc = accuracy_score(y_true=targets, y_pred=np.where(preds >= 0.5, 1, 0)) # 여기서 잘못된 듯 
    # print(f'get_metric 내부 === auc_score: {preds[10:20]} ======')
    # print(f'get_metric 내부 === acc_pred: {np.where(preds >= 0.5, 1, 0)[10:20]} ======')
    # print(f'get_metric 내부 === auc: {auc}, acc: {acc} ======') # 여기서도 업데이트 안됨 
    return auc, acc
