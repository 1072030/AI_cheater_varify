'''
這個程式是randomForest classifier的預測分析
由於執行效果不佳，不適用
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    # optimize
    n_estimators = [200,300,400,500]
    max_depth= [2,3,4,5,6]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    param_grid = dict(n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf)

    grid_search = GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
    best_clf = grid_search
    best_clf.fit(X_train, y_train)

    print("best parameters:", grid_search.best_params_)
    print("best grade:", grid_search.best_score_)

    print("RandomForest model test score",best_clf.score(X_test, y_test))
    y_predict = best_clf.predict(X_test)
    print("RandomForest predict-----")
    print("predicted",y_predict)
    # print("accuracy",accuracy_score(y_test, y_predict))

    y_predict = best_clf.predict(predict_X)
    print("RandomForest predict-----")
    print("predicted",y_predict)
    predict_set.insert(loc=predict_set.shape[1],column="predict_SVM",value=y_predict.tolist())
    predict_set.to_excel(os.getcwd()+'/RF_file/RandomForest_data_analyze_2.xlsx')

SVM_data_analyze(df_use)