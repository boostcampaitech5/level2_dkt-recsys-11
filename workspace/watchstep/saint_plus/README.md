# SAINT+

## 🔨 Setting up the Environment
```bash
cd /opt/ml/input/code/saint_plus
conda init
(base) . ~/.bashrc
(base) conda create -n saint-plus python=3.10
(base) conda activate saint-plus
(saint-plus) pip install -r requirements_short.txt
(saint-plus) python train.py
(saint-plus) python inference.py
```

## 📂 Files
`code/saint_plus`
* `train.py`: 학습코드
* `inference.py`: 추론 후 `submissions.csv` 파일을 만들어주는 소스코드
* `requirements_short.txt`: 모델 학습에 필요한 라이브러리들이 버전 명시없이 정리되어 있음.

`code/saint_plus/sain_plus`
* `config.py`: 학습에 활용되는 여러 argument들을 적은 config 파일.
* `datasets.py`: 학습 데이터를 불러 SAINT+ 입력에 맞게 변환.
* `trainer.py`: 훈련에 사용되는 함수들을 포함.
* `utils.py`: 학습에 필요한 부수적인 함수들을 포함.