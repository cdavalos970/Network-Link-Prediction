import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV

file_link_train = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/dataFrame_analysis_f.csv'
file_link_test = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/dataFrame_analysis_test.csv'

file_output = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/result_SVM_CV_HE_R.csv'

df_train = pd.read_csv(file_link_train) 
df_train.resour = np.log(df_train.resour + 0.0005)
neigh_encode = pd.get_dummies(pd.cut(df_train.neighbors, bins=list({0,1,2,3,100})))

df_final = pd.read_csv(file_link_test) 
df_final.resour = np.log(df_final.resour + 0.0005)
neigh_encode_final = pd.get_dummies(pd.cut(df_final.neighbors, bins=list({0,1,2,3,100})))

X = df_train.drop(columns=['Node', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors'])
y = df_train[['Node']]
X_final = df_final.drop(columns=['Id', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors'])
y_final = df_final[['Id']]

scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))
X_final = pd.DataFrame(scaler.transform(X_final))
X = pd.concat([X, neigh_encode.reset_index(drop=True)], axis=1)
X_final = pd.concat([X_final, neigh_encode_final.reset_index(drop=True)], axis=1)

parameter_space = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svmm = svm.SVC()
clf = GridSearchCV(svmm, parameter_space, n_jobs = 3, scoring = 'roc_auc')
search = clf.fit(X, y.values.ravel())
print('Best Params')
print(search.best_params_)
print('Results')
print(search.cv_results_)

y_pred_test = clf.predict_proba(X_final)[:,1].round(decimals=6)
df_test_LR = pd.DataFrame({'Id': y_final.values.ravel(), 'Predicted': np.array(y_pred_test)})
df_test_LR.to_csv(file_output, index = False, header=True)
