import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    # 디렉토리 및 파일명 지정 
    parser.add_argument("--data_dir", default="/opt/ml/input/data/", type=str, help="data directory")
    parser.add_argument("--asset_dir", default="asset/", type=str, help="data directory")
    parser.add_argument("--file_name", default="train_data.csv", type=str, help="train file name")
    parser.add_argument("--model_dir", default="models/", type=str, help="model directory")
    parser.add_argument("--model_name", default="best_model.pt", type=str, help="model file name")
    parser.add_argument("--output_dir", default="outputs/", type=str, help="output directory")
    parser.add_argument("--test_file_name", default="test_data.csv", type=str, help="test file name")
    parser.add_argument("--memo", default='', type=str, help="for wandb name memo") # wandb run name 

    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument("--max_seq_len", default=64, type=int, help="max sequence length")


    # 모델
    parser.add_argument("--hidden_dim", default=64, type=int, help="hidden dimension size")
    parser.add_argument("--n_layers", default=1, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.3, type=float, help="drop out rate")
    parser.add_argument("--dim_div", default=3, type=int, help="divider for hidden dimensions")
    

    # 훈련
    parser.add_argument("--n_epochs", default=60, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")
    parser.add_argument("--log_steps", default=50, type=int, help="print log per n steps")
    parser.add_argument("--use_test_data", default=False, type=bool, help="use test data for training") # w/ test data


    # 중요
    parser.add_argument("--model", default="lstmattn", type=str, help="model type")
    parser.add_argument("--optimizer", default="adamW", type=str, help="optimizer type")
    parser.add_argument("--scheduler", default="plateau", type=str, help="scheduler type")


    # K-Fold
    parser.add_argument("--use_kfold", default=False, type=bool, help="use k fold cross validation")
    parser.add_argument("--kfold", default=10, type=int, help="num of k fold")


    args = parser.parse_args()

    return args
