# Deep Knowledge Tracing
> 개개인에 맞춤화된 지식 상태를 추적하는 딥러닝 프로젝트  
> (2023-05-03 ~ 2023-05-25)

<br>
<div align="center">
<img src="https://img.shields.io/badge/Python-3776AB?logo=Python&logoColor=white" alt="Python badge">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=PyTorch&logoColor=white" alt="PyTorch badge">
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="pandas badge">
  <img src="https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white" alt="numpy badge">
   <img src="https://img.shields.io/badge/scikit learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikitlearn badge">
  <img src="https://img.shields.io/badge/wandb-FFBE00?logo=weightsandbiases&logoColor=white" alt="weightsandbiases badge">
 <img src="https://img.shields.io/badge/-Sweep-orange" alt="scikitlearn badge">
  <img src="https://img.shields.io/badge/-Optuna-blue" alt="optuna badge">
</div>


## Members

<div align="center">
<table>
  <tr>
     <td align="center">
        <a href="https://github.com/gangjoohyeong">
          <img src="https://avatars.githubusercontent.com/u/93419379?v=4" width="100px" alt=""/><br />
          <sub><b>강주형</b></sub>
        </a><br/>
    </td>
    <td align="center">
        <a href="https://github.com/watchstep">
          <img src="https://avatars.githubusercontent.com/u/88659167?v=4" width="100px" alt=""/><br />
          <sub><b>김주의</b></sub>
        </a><br/>
    </td>
    <td align="center">
        <a href="https://github.com/eunjios">
          <img src="https://avatars.githubusercontent.com/u/77034159?v=4" width="100px" alt=""/><br />
          <sub><b>이은지</b></sub>
        </a><br/>
    </td>
    <td align="center">
        <a href="https://github.com/hoyajigi">
          <img src="https://avatars.githubusercontent.com/u/1335881?v=4" width="100px" alt=""/><br />
          <sub><b>조현석</b></sub>
        </a><br/>
    </td>
    <td align="center">
        <a href="https://github.com/MSGitt">
          <img src="https://avatars.githubusercontent.com/u/121923924?v=4" width="100px" alt=""/><br />
          <sub><b>최민수</b></sub><br/>
        </a>
    </td>
  </tr>
</table>

| 공통 | EDA, Data preprocessing, Feature engineering |
| :---: | :--- |
| 강주형 | ML modeling, Graph modeling, Hyper parameter tuning, Ensemble |
| 김주의 | SAINT+ modeling |
| 이은지 | DL modeling, Graph model tuning, K-fold |
| 조현석 | Hyper parameter tuning, Ensemble, Manage data pipeline |
| 최민수 | ML modeling, Hyper parameter tuning, Schedule management, Ensemble |
</div>

## 팀 목표
1. 팀원 모두가 머신러닝 전체 프로세스에 대해 참여하고 이해하기  
2. 주어진 Sequnetial 데이터의 특성을 고려해서 모델링 진행하기  
3. 모델 성능 향상뿐만 아니라 학습적인 측면도 고려해서 진행하기  
4. 생산성을 높일 수 있는 다양한 Tool을 도입해 보기  

## 모델
<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align:center">Weighted Ensemble</th>
    </tr>
    <tr>
      <th style="text-align:center">🦾 Model</th>
      <th style="text-align:center">⚖️ Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">XGBoost</td>
      <td style="text-align:center">0.6</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align:center">LightGBM</td>
      <td style="text-align:center">0.2</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align:center">GRUATTN</td>
      <td style="text-align:center">0.1</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align:center">LightGCN</td>
      <td style="text-align:center">0.1</td>
    </tr>
  </tbody>
</table>


||🔒 Private|🔑 Public|
|:---:|:---:|:---:|
|AUROC|0.8549|0.8225|


## 데이터셋 구조

```
data/
├── sample_submission.csv
├── test_data.csv
└── train_data.csv
```



<br>
<div align="center"><a href="https://github.com/boostcampaitech5/level2_dkt-recsys-11"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/boostcampaitech5/level2_dkt-recsys-11&count_bg=%23FF7474&title_bg=%23212020&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a></div>
