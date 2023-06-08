'''
進行logisticRegression 的程式碼
將檔案輸入至 /logistic_file/
'''

import os
import sys
import pandas as pd
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import math
from time import perf_counter
start_reading = perf_counter()
print("Start reading...")
print("useful data...")
df_use = pd.read_excel(os.getcwd()+'/summary_data/useful_for_train.xlsx')

end_reading = perf_counter()
print("End of reading...")
print("-------------------------------------------------------------")
print(f"reading process cost Time : {format(end_reading-start_reading)}")
print("-------------------------------------------------------------")

def SVM_data_analyze(summary:pd.DataFrame):
    # 決策樹
    # 常見的決策亂度評估指標有 Information gain、Gain ratio、Gini index。
    #Parameters:
    # criterion: 亂度的評估標準，gini/entropy。預設為gini。
    # max_depth: 樹的最大深度。
    # splitter: 特徵劃分點選擇標準，best/random。預設為best。
    # random_state: 亂數種子，確保每次訓練結果都一樣，splitter=random 才有用。
    # min_samples_split: 至少有多少資料才能再分
    # min_samples_leaf: 分完至少有多少資料才能分
    # print("X_train",X_train.shape,X_train) (79 rows 5 col)
    # print("X_test",X_test.shape,X_test) (20 rows 5 col)
    # print("y_train",y_train.shape,y_train) (79 rows 1 col)
    # print("y_test",y_test.shape,y_test) (20rows 1 col)

   

    dataset = summary[summary["cheater"] != 2]
    predict_set = summary[summary["cheater"] == 2]
    predict_X = predict_set.drop(['cheater','userId'],axis=1)

    y = dataset["cheater"]
    X = dataset.drop(['cheater','userId'],axis=1)
    print("trainingset_size",X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # optimize
    penalty = ['l1','l2']
    C= [0.001,0.01,0.1,1,10]


    param_grid = dict(penalty=penalty,
                      C=C)

    grid_search = GridSearchCV(LogisticRegression(),param_grid,cv=5)
    best_clf = grid_search
    best_clf.fit(X_train, y_train)

    print("best parameters:", grid_search.best_params_)
    print("best grade:", grid_search.best_score_)

    print("logisticRegression model test score",best_clf.score(X_test, y_test))
    y_predict = best_clf.predict(X_test)
    print("logisticRegression predict-----")
    print("predicted",y_predict)
    cm = confusion_matrix(y_test, y_predict)
    plt.imshow(cm,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center')
    plt.xlabel("logisticRegression label")
    plt.ylabel("truth label")
    plt.show()
    # print("accuracy",accuracy_score(y_test, y_predict))

    y_predict = best_clf.predict(predict_X)
    print("logisticRegression predict-----")
    print("predicted",y_predict)
    predict_set.insert(loc=predict_set.shape[1],column="predict_SVM",value=y_predict.tolist())
    predict_set.to_excel(os.getcwd()+'/logistic_file/logisticRegression_1.xlsx')

SVM_data_analyze(df_use)