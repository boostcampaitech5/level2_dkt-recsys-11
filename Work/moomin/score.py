import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
answer_data = pd.read_csv('maybe_answer.csv')
pred_data = pd.read_csv('output.csv')
auc = roc_auc_score(y_true=answer_data["prediction"], y_score=pred_data["prediction"])
acc = accuracy_score(y_true=answer_data["prediction"], y_pred=pred_data["prediction"] > 0.5)
print("AUC : %.4f ACC : %.4f" % (auc, acc))