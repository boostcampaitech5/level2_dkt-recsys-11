import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self,
                   data: np.ndarray,
                   ratio: float = 0.9, # ====== 0.7 -> 0.9 로 수정
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        return data_1, data_2

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        # ========== ADD: 여기 부분만 바꾸면 됨 -> args 로 받아오도록 ================
        self.args.columns                           # df의 모든 columns 저장된 리스트 
        self.args.user_cols = ['userID']            # 고정
        self.args.target_cols = ['answerCode']      # 고정
        self.args.drop_cols = ['Timestamp', 'time_diff', 'days_diff'] 
        self.args.cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']
        self.args.graph_cols = ['assessmentItemID', 'testId', 'KnowledgeTag'] # 그래프 임베딩 columns
        drop = self.args.user_cols + self.args.target_cols + self.args.drop_cols + self.args.cate_cols
        self.args.cont_cols = [x for x in self.args.columns if x not in drop]

        # TODO: assert
        print(f'columns: {self.args.columns}')
        print(f'continuous cols: {self.args.cont_cols}')
        # ====================================================================

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        # label encoding 
        for col in self.args.cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )
            # self.args.cate_cols 에 포함되는 컬럼이 범주형 
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        time_diff: datetime64[ns]
        days_diff: float
        seconds_diff: float
        log_days_diff: float
        correct_by_diff: float
        correct_by_tag: float
        """
        # TODO: FE 추가하기 (지금은 대충함)
        df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        
        # Timestamp의 hour 정보로 binning
        # df['timestamp_bin'] = df['Timestamp'].transform(lambda x: x.hour)

        # 이전 문제와 현재 문제 timestamp의 차이 
        # df['time_shift_1'] = df.groupby('userID')['Timestamp'].shift(1)
        df['time_diff'] = df['Timestamp'] - df.groupby('userID')['Timestamp'].shift(1)
        df['days_diff'] = df['time_diff'].map(lambda x: x.days)

        # seconds_diff: 현재 문제 푸는데 걸린 시간 
        df['seconds_diff'] = df['time_diff'].map(lambda x: x.seconds)
        mask = (df['days_diff'] > 0) | ((df['days_diff'] == 0) & (df['seconds_diff'] > 600))
        df.loc[mask, 'seconds_diff'] = np.nan    # 하루 이상 걸린 경우 reset
        df['seconds_diff'].fillna(df['seconds_diff'].median(), inplace=True)

        # skewed -> log transformation
        df['log_days_diff'] = np.log1p(df['days_diff'])

        # 유저별 seconds_diff 별 정답률
        df['correct_by_diff'] = df.groupby(['userID', 'seconds_diff'])['answerCode'].transform('mean')
        
        # 유저별 tag 별 정답률
        df['correct_by_tag'] = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].transform('mean')
        
        # ======== ADD: 모든 컬럼을 args.columns 에 추가 ========
        self.args.columns = df.columns.tolist()
        # ==================================================
        return df
    
    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # TODO: args.use_test_data == True -> test 도 학습에 사용 

        # ======================== ADD: userID 개수(graph) =====================
        self.args.n_userID = df['userID'].nunique()
        # =====================================================================

        # ======================= ADD: nunique + 1 값 지정 ======================
        for col in self.args.cate_cols:
            setattr(self.args, f'n_{col}', len(
                np.load(os.path.join(self.args.asset_dir, f'{col}_classes.npy'))
            ))
        # =====================================================================

  
        # ============== ADD : 사용되는 컬럼 인자로 받기 + group 코드 변경 ================
        columns = self.args.cate_cols + self.args.cont_cols + self.args.target_cols

        group = df[columns + self.args.user_cols].groupby("userID").apply(
            lambda r: tuple([r[col].values for col in columns])
        )
        # =========================================================================

        return group.values

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name) # return group.values

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        # ========= ADD: args 추가 ============
        self.args = args
        # ====================================
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index] 
        # row: index 번째 유저 정보 (feature1의 seq, featrue2의 seq, ...)

        # =================== ADD: 데이터 불러오기 ====================
        data = {}
        columns = self.args.cate_cols + self.args.cont_cols + self.args.target_cols # 이 순서

        for i, col in enumerate(columns):
            if i < len(self.args.cate_cols): # categorical
                data[col] = torch.tensor(row[i] + 1, dtype=torch.int)
            else: # continous
                data[col] = torch.tensor(row[i], dtype=torch.float)
        # =========================================================

        # mask: max_seq_len 기준으로 길면 자르고, 짧으면 pre-padding 
        seq_len = len(row[0]) # 
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len-seq_len:] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask
        
        # Generate interaction
        interaction = data[self.args.target_cols[0]] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        
        return data

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader
