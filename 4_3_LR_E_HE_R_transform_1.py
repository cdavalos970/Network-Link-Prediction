import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

file_link_train = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/dataFrame_analysis_f_1.csv'
file_link_test = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/dataFrame_analysis_test.csv'

file_output = '/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/result_LR_CV_HE_R_transform_f.csv'

df_train = pd.read_csv(file_link_train) 
df_train_transform_pow_2 = df_train.drop(columns=['Node', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last']).transform([lambda x : x**2]).add_suffix('_pow_2')
df_train_transform_pow_3 = df_train.drop(columns=['Node', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last']).transform([lambda x : x**3]).add_suffix('_pow_3')
df_train_transform_log = df_train.drop(columns=['Node', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last']).transform([lambda x : np.log(x + 0.0001)]).add_suffix('_log')
df_train_transform_pow_4 = df_train.drop(columns=['Node', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last']).transform([lambda x : x**4]).add_suffix('_pow_4')

neigh_encode = pd.get_dummies(pd.cut(df_train.neighbors, bins=list({0,1,2,3,100})))
dif_last_encode = pd.get_dummies(df_train.dif_last)

df_final = pd.read_csv(file_link_test) 
df_final_transform_pow_2 = df_final.drop(columns=['Id', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last']).transform([lambda x : x**2]).add_suffix('_pow_2')
df_final_transform_pow_3 = df_final.drop(columns=['Id', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last']).transform([lambda x : x**3]).add_suffix('_pow_3')
df_final_transform_log = df_final.drop(columns=['Id', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last']).transform([lambda x : np.log(x + 0.0001)]).add_suffix('_log')
df_final_transform_pow_4 = df_final.drop(columns=['Id', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last']).transform([lambda x : x**4]).add_suffix('_pow_4')

neigh_encode_final = pd.get_dummies(pd.cut(df_final.neighbors, bins=list({0,1,2,3,100})))
dif_last_encode_final = pd.get_dummies(df_final.dif_last)

X = df_train.drop(columns=['Node', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last'])
X = pd.concat([X, df_train_transform_pow_2.reset_index(drop=True)], axis=1)
X = pd.concat([X, df_train_transform_pow_3.reset_index(drop=True)], axis=1)
X = pd.concat([X, df_train_transform_log.reset_index(drop=True)], axis=1)
X = pd.concat([X, df_train_transform_pow_4.reset_index(drop=True)], axis=1)

y = df_train[['Node']]
X_final = df_final.drop(columns=['Id', 'Sink', 'Source', 'path_num', 'sim', 'short', 'dif_rate', 'neighbors', 'dif_last'])
X_final = pd.concat([X_final, df_final_transform_pow_2.reset_index(drop=True)], axis=1)
X_final = pd.concat([X_final, df_final_transform_pow_3.reset_index(drop=True)], axis=1)
X_final = pd.concat([X_final, df_final_transform_log.reset_index(drop=True)], axis=1)
X_final = pd.concat([X_final, df_final_transform_pow_4.reset_index(drop=True)], axis=1)

y_final = df_final[['Id']]

pd.set_option('max_columns', None)

scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))
X_final = pd.DataFrame(scaler.transform(X_final))

X = pd.concat([X, neigh_encode.reset_index(drop=True)], axis=1)
X = pd.concat([X, dif_last_encode.reset_index(drop=True)], axis=1)

X_final = pd.concat([X_final, neigh_encode_final.reset_index(drop=True)], axis=1)
X_final = pd.concat([X_final, dif_last_encode_final.reset_index(drop=True)], axis=1)


parameter_space = {'C': uniform(0.0001, 0.9),
    'penalty':['l2', 'l1', 'elasticnet'],
    'solver': ['lbfgs', 'liblinear', 'saga']}

nn = LogisticRegression(max_iter = 10000)
clf = RandomizedSearchCV(nn, parameter_space, random_state=0, n_jobs = 3, scoring = 'roc_auc', n_iter = 50, cv = 20)
search = clf.fit(X, y.values.ravel())
print('Best Params')
print(search.best_params_)
print('Results')
print(search.cv_results_)

y_pred_test = clf.predict_proba(X_final)[:,1].round(decimals=6)
df_test_LR = pd.DataFrame({'Id': y_final.values.ravel(), 'Predicted': np.array(y_pred_test)})
df_test_LR.to_csv(file_output, index = False, header=True)

