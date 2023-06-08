'''
這是將同學分析過後的資料進行整合
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
df_use = pd.read_excel(os.getcwd()+'/summary_data/useful.xlsx')
df_all = pd.read_csv(os.getcwd()+"/summary_data/3 June_df final_game cheater.csv")

end_reading = perf_counter()
print("End of reading...")
print("-------------------------------------------------------------")
print(f"reading process cost Time : {format(end_reading-start_reading)}")
print("-------------------------------------------------------------")

# print(df_cheater[df_cheater["UserID"]=="temp000083"]["DailyAvgTransactions"].values[0])
# print(df_cheater["UserID"].values)
TotalTransactions = []
DailyAvgTransactions = []
print("my dataset size",len(df_use["userId"].values))
for i in df_use["userId"]:
    # print(i)
    if i in df_all["UserID"].values:
        TotalTransactions.append(df_all[df_all["UserID"]==i]["TotalTransactions"].values[0])
        DailyAvgTransactions.append(df_all[df_all["UserID"]==i]["DailyAvgTransactions"].values[0])
   
print("match size",DailyAvgTransactions.__len__())
print("match size",TotalTransactions.__len__())