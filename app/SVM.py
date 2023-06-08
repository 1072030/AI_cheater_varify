'''
這個程式是負責進行SVM classifier
將預測完的結果輸入至 /SVM_file/
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
    dataset = summary[summary["cheater"] != 2]
    predict_set = summary[summary["cheater"] == 2]
    predict_X = predict_set.drop(['cheater','userId'],axis=1)

    y = dataset["cheater"]
    X = dataset.drop(['cheater','userId'],axis=1)
    print("trainingset_size",X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    # ---- GridSearch parameter
    C = [0.01,0.05,0.1]
    kernel = ['linear', 'rbf', 'poly']
    gamma= [0.01,0.05,0.1]
    # ---- GridSearch parameter
    param_grid = dict(C=C,kernel=kernel,gamma=gamma)
    grid_search = GridSearchCV(svm.SVC(),param_grid,cv=10)
    best_clf = grid_search
    best_clf.fit(X_train, y_train)

    print("best parameters:", grid_search.best_params_)
    print("best grade:", grid_search.best_score_)

    print("SVM model test score",best_clf.score(X_test, y_test))
    y_predict = best_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    plt.imshow(cm,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center')
    plt.xlabel("SVM label")
    plt.ylabel("truth label")
    plt.show()

    y_predict = best_clf.predict(predict_X)
    print("SVM predict-----")
    print("predicted",y_predict)
    predict_set.insert(loc=predict_set.shape[1],column="predict_SVM",value=y_predict.tolist())
    predict_set.to_excel(os.getcwd()+'/SVM_file/SVM_5.xlsx')

SVM_data_analyze(df_use)