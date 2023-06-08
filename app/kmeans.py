

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
# print("reading Catching AI total file")
# df_Catch_total = pd.read_excel(os.getcwd()+'/轉轉樂啟動次數記錄.xlsx',usecols=["UserID","Times"])
print("reading NormalPerson file...")
df_Normal = pd.read_excel(os.getcwd()+'/亂抓外掛Log_1000正常人.xlsx',usecols=["UserID","Times"],sheet_name="轉轉樂啟動次數記錄")
print("reading Cheater file...")
df_Cheater = pd.read_excel(os.getcwd()+'/cheaters.xlsx',usecols=["UserID"])
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

def cal_avg(summary:dict):
    userId = []
    avg_times = [] # 除以天數
    timediff_mid=[] # 除以數量
    totoal_times=[]
    standard_deviation=[]
    date =[]
    for i in summary.keys():
        sum_times=0
        sum_timediff_mid=0
        standard_deviation_sum = []
        days = summary[i].keys().__len__()
        for j in summary[i].keys():
            sum_times+=int(summary[i][j]["times"])

            for k in summary[i][j]["timediff"]:
                standard_deviation_sum.append(k)

            Sum = sum(summary[i][j]["timediff"])
            sum_timediff_mid += Sum/len(summary[i][j]["timediff"])
        

        standard_deviation.append(np.std(standard_deviation_sum))
        totoal_times.append(sum_times)
        sum_times = sum_times/days
        sum_timediff_mid = sum_timediff_mid/days
        userId.append(i)
        avg_times.append(sum_times)
        timediff_mid.append(sum_timediff_mid)
        date.append(days)
        
    data = {
            "userId":userId,
            "avg_times":avg_times,
            "timediff_mid":timediff_mid,
            "total_times":totoal_times,
            "standard_deviation":standard_deviation,
            "login_days":date
        }
    df = pd.DataFrame(data)
    return df,summary

def cheater_define(summary:pd.DataFrame):
    normal = []
    cheater = []
    summary_total = []

    for i in df_Normal.iterrows():
        temp = i[1].values
        userId = temp[0]
        normal.append(userId)
    for i in df_Cheater.iterrows():
        temp = i[1].values
        userId = temp[0]
        cheater.append(userId)

    for i in summary.iterrows():
        temp = i[1].values
        userId = temp[0]
        if userId in normal:
            summary_total.append(0)
        elif userId in cheater:
            summary_total.append(1)
        else:
            summary_total.append(2)

    print("after",summary.info)
    print("summary_total",summary_total.__len__())
    summary.insert(loc=summary.shape[1],column="cheater",value=summary_total)
    return summary

def write_pic(summary:dict):
    avg_daily_timediff={}
    name = "jonathan7958"
    for i in summary.keys():
        # days = len(summary[i].keys())
        # print(i)
        if i == name:
            print(i)
            for j in summary[i].keys():
                count = 0
                count += sum(summary[i][j]["timediff"])
                if j not in avg_daily_timediff.keys():
                    avg_daily_timediff[j] = avg_daily_timediff.get(j,count/len(summary[i][j]["timediff"]))
            
    print("")
    date = avg_daily_timediff.keys()
    timediff = avg_daily_timediff.values()
    print("date",date.__len__())
    print(date)
    print("timediff",timediff.__len__())
    print(timediff)
    data = {
        "date":date,
        "timediff":timediff
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(6,4))
    plt.title(label=f"userId:{name}")
    ax=sns.lineplot(data = df, x=f"date", y="timediff")
    plt.savefig(os.getcwd()+f'/timediff_pic/{name}_{date.__len__()}days.png')
    plt.show()
# def kmeans(summary:pd.DataFrame):
# m0000741

def kmeans_pca_analyze(summary:pd.DataFrame):
    X = summary.drop(['userId',"timediff_mid"],axis=1)
    # minScaler = MinMaxScaler()
    # x_minMax = minScaler.fit_transform(X)
    pca = PCA(n_components=2)
    pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(data = pca,columns = ['pc1', 'pc2'])

    dataset = list(zip(pca_df["pc1"],pca_df["pc2"]))
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit(dataset)
    y = kmeans.predict(dataset)
    pca_df["userId"] = summary["userId"]
    pca_df["target"] = y
    print(y)
    colors = ['r', 'g']
    y = y.tolist()
    for i in range(len(y)):
        plt.scatter(pca_df["pc1"][i], pca_df["pc2"][i], color=colors[y[i]], s=50)
    plt.title("k-means")
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.savefig(os.getcwd()+f'/kmeans_output.png')
    plt.show()
    print("y_predict")
    print(y)
    print("pca_df")
    print(pca_df.head())
    pca_df.to_excel(os.getcwd()+'/kmeans_file/kmeans_without_timediffmid.xlsx')
    return pca_df

print("Starting process...")
summary = analyze(df_Catch) # 正常人的timediff 和 times
write_pic(summary)

# print(pic)
df,summary = cal_avg(summary)
print(df.head())
print(df.shape)
pca_df = kmeans_pca_analyze(df)

cheater=[]
for i in df_Cheater.iterrows():
        temp = i[1].values
        userId = temp[0]
        cheater.append(userId)
not_incheater=[]
print(cheater)
pca_df = pca_df[pca_df["target"]==1]
for i in pca_df["userId"]:

    if i not in cheater:
        not_incheater.append(i)
print(not_incheater)
print(not_incheater.__len__())

# df.to_excel(os.getcwd()+'/summary_data/test.xlsx')
# df = cheater_define(df)
# print(df.head())
# print(df.shape)
# df.to_excel(os.getcwd()+'/summary_data/useful.xlsx')
# df = df[df["cheater"] != 2]
# df.to_excel(os.getcwd()+'/summary_data/useful_without.xlsx')






