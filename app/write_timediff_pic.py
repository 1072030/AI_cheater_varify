'''
這個程式主要是負責呈現資料中的timediff呈現狀況並將圖片儲存下來
執行時需要一個固定參數                      ↓是userId
Example: python app/write_timediff_pic.py a0985010777
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
print("reading Catching AI file...")
df_Catch = pd.read_excel(os.getcwd()+'/120秒內解除轉轉樂紀錄.xlsx',usecols=["UserID","fireTime","timediff"])
end_reading = perf_counter()
print("End of reading...")
print("-------------------------------------------------------------")
print(f"reading process cost Time : {format(end_reading-start_reading)}")
print("-------------------------------------------------------------")

def analyze(total:pd.DataFrame):
    summary={}
    for i in total.iterrows():
        temp = i[1].values
        userId = temp[0]
        date = datetime.datetime.strptime(temp[1],"%Y-%m-%d %H:%M:%S.%f")
        timediff = temp[2]
        if userId not in summary.keys():
            summary[userId] = summary.get(userId,{})

        day_str="0"
        date_for_keys=0
        if date.day in range(1,10):
            day_str = day_str + str(date.day)
            date_for_keys = int(str(date.month)+day_str)
        else:
            date_for_keys = int(str(date.month)+str(date.day))
        
        if date_for_keys not in summary[userId].keys():
            summary[userId][date_for_keys] = summary.get(date_for_keys,{"times":0,"timediff":[]})

        summary[userId][date_for_keys]["timediff"].append(timediff)
        summary[userId][date_for_keys]["times"]+=1
    return summary

def write_pic(summary:dict):
    avg_daily_timediff={}
    name = sys.argv[1]
    for i in summary.keys():
        if i == name:
            print(i)
            for j in summary[i].keys():
                print(summary[i][j]["timediff"])
                count = 0
                count += sum(summary[i][j]["timediff"])
                if j not in avg_daily_timediff.keys():
                    avg_daily_timediff[j] = avg_daily_timediff.get(j, count/len(summary[i][j]["timediff"]))
            

    date = list(avg_daily_timediff.keys())
    timediff = list(avg_daily_timediff.values())
    summary_data = {}
    for i in range(len(date)):
        summary_data[date[i]] = timediff[i]
    summary_data = dict(sorted(summary_data.items()))
    print("summary_data",summary_data)
    print("date",date.__len__())
    print(date)
    print("timediff",timediff.__len__())
    print(timediff)
    data = {
        "date":summary_data.keys(),
        "timediff":summary_data.values()
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(10,6))
    plt.title(label=f"userId:{name}")
    ax=sns.lineplot(data = df, x=f"date", y="timediff")
    ax.set_xticklabels(labels=summary_data.keys(), rotation=90, ha='right')
    plt.savefig(os.getcwd()+f'/timediff_pic/{name}_{date.__len__()}days.png')
    plt.show()
# def kmeans(summary:pd.DataFrame):
# m0000741

print("Starting process...")
summary = analyze(df_Catch) # 正常人的timediff 和 times
write_pic(summary)








