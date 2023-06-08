'''
這個程式主要是用來將每個模型所產出的cheater進行整合
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
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.cluster import DBSCAN

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import math
from time import perf_counter


start_reading = perf_counter()
print("Start reading...")

print("reading Decision file...")
df_Decision = pd.read_excel(os.getcwd()+'/decision_file/decision_final.xlsx')
print(df_Decision.head(1))
print("reading Kmeans file...")
df_Kmeans = pd.read_excel(os.getcwd()+'/kmeans_file/kmeans_without_timediffmid.xlsx')
print(df_Kmeans.head(1))
print("reading Logistic file...")
df_Logistic = pd.read_excel(os.getcwd()+'/logistic_file/logisticRegression_1.xlsx')
print(df_Logistic.head(1))
print("reading SVM file...")
df_SVM = pd.read_excel(os.getcwd()+'/SVM_file/SVM_5.xlsx')
print(df_SVM.head(1))
print("reading Cheater file...")
df_cheater = pd.read_excel(os.getcwd()+'/cheaters.xlsx',usecols=["UserID"])
print("useful data...")
df_use = pd.read_excel(os.getcwd()+'/summary_data/dataset_Mike.xlsx')
print("reading NormalPerson file...")
df_Normal = pd.read_excel(os.getcwd()+'/亂抓外掛Log_1000正常人.xlsx',usecols=["UserID","Times"],sheet_name="轉轉樂啟動次數記錄")
end_reading = perf_counter()



beyond_cheater_file = []
cheater = []
normal = []
for i in df_cheater["UserID"]:
    if i not in cheater:
        cheater.append(i)
# for i in df_Normal["UserID"]:
#     if i not in normal:
#         normal.append(i)
def check(summary:pd.DataFrame,ping:int=2):
    for i in summary.iterrows():
        temp = i[1].values
        userId = temp[ping]
        target = temp[len(temp)-1]
        print("target",userId,target)
        if userId not in cheater and userId not in normal and  target == 1:
            beyond_cheater_file.append(userId)

check(df_Decision)
check(df_Logistic)
check(df_SVM)
check(df_Kmeans,1)

print(beyond_cheater_file)
print(len(beyond_cheater_file))

userId=[]
avg_times_per_day=[]
timediff_mid=[]
total_times=[]
standard_deviation=[]
login_days=[]
for i in df_use["userId"]:
    if i in beyond_cheater_file:
        userId.append(i)
        avg_times_per_day.append(df_use[df_use["userId"]==i]["avg_times_per_day"].values[0])
        timediff_mid.append(df_use[df_use["userId"]==i]["timediff_mid"].values[0])
        total_times.append(df_use[df_use["userId"]==i]["total_times"].values[0])
        standard_deviation.append(df_use[df_use["userId"]==i]["standard_deviation"].values[0])
        login_days.append(df_use[df_use["userId"]==i]["login_days"].values[0])

data = {
    "userId":userId,
    "avg_times_per_day":avg_times_per_day,
    "timediff_mid":timediff_mid,
    "total_times":total_times,
    "standard_deviation":standard_deviation,
    "login_days":login_days,
}
df = pd.DataFrame(data)
df.to_excel(os.getcwd()+'/final_output_with_remove.xlsx')
