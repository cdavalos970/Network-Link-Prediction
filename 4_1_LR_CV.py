import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

file_link_train = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/dataFrame_analysis_f.csv'
file_link_test = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/dataFrame_analysis_test.csv'

file_output = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/result_LR_CV.csv'

df_train = pd.read_csv(file_link_train) 

df_final = pd.read_csv(file_link_test) 

X = df_train.drop(columns=['Node', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate'])
y = df_train[['Node']]
X_final = df_final.drop(columns=['Id', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate'])
y_final = df_final[['Id']]

parameter_space = {'C': uniform(0.0001, 0.9),
    'penalty':['l2', 'l1', 'elasticnet'],
    'solver': ['lbfgs', 'liblinear', 'saga']}

nn = LogisticRegression(max_iter = 2000)
clf = RandomizedSearchCV(nn, parameter_space, random_state=0, n_jobs = 3, scoring = 'roc_auc', n_iter = 30, cv = 10)
search = clf.fit(X, y.values.ravel())
print('Best Params')
print(search.best_params_)
print('Results')
print(search.cv_results_)

y_pred_test = clf.predict_proba(X_final)[:,1].round(decimals=6)
df_test_LR = pd.DataFrame({'Id': y_final.values.ravel(), 'Predicted': np.array(y_pred_test)})
df_test_LR.to_csv(file_output, index = False, header=True)
