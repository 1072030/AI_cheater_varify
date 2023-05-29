import os
import sys
import pandas as pd
import numpy as np

import sklearn
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
from sklearn import svm

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import math
from time import perf_counter
# dir_path = os.path.dirname(os.path.realpath(__file__)) C:\Users\ASUS\Desktop\code\cheater_analyze\app
# os.getcwd() C:\Users\ASUS\Desktop\code\cheater_analyze
# sheet_name list ["一億以上亂幣交易紀錄","賣商店次數前100名角色","賣商店次數前100名角色1月賣商店紀錄","賣商店次數前100名角色2月賣商店紀錄",
#                  "賣商店次數前100名角色3月賣商店紀錄","120秒內解除轉轉樂紀錄","轉轉樂啟動次數記錄"]
# --------------------------------------------------------------
# key data needed  usecols=[]
# 賣商店次數前100名角色             : ["UserID","SoldCount"]
# 賣商店次數前100名角色1月賣商店紀錄 : ["UserID","ExchangeDate"]
# 120秒內解除轉轉樂紀錄             : ["UserID","fireTime","timediff"]
# 轉轉樂啟動次數記錄                : ["UserID","Times"]
# --------------------------------------------------------------
# What are we search for ?
# Y = f(x)
# Y 可以是 boolean 是否為外掛
# Y 也可以是 此使用者是外掛的機率
# 難度差異蠻大
# x 
# 玩家的一天的登入時間 firetime
# 解除轉轉樂時間 timediff standard deviation
# 賣商店的次數 ExchangeRecord
kmans_path = os.getcwd()+'/kmeans/'
regression_path = os.getcwd()+'/regression/'
decision_path = os.getcwd()+'/decision/'
# --------------------------------------------------------------
# print(sys.argv)
start_reading = perf_counter()
print("Start reading...")
print("reading Jan file...")
df_Jan = pd.read_excel(os.getcwd()+'/賣商店次數前100名角色1月賣商店紀錄.xlsx',usecols=["UserID","ExchangeDate"])
print("reading Feb file...")
df_Feb = pd.read_excel(os.getcwd()+'/賣商店次數前100名角色2月賣商店紀錄.xlsx',usecols=["UserID","ExchangeDate"])
print("reading March file...")
df_March = pd.read_excel(os.getcwd()+'/賣商店次數前100名角色3月賣商店紀錄.xlsx',usecols=["UserID","ExchangeDate"])
print("reading Catching AI file")
df_Catch = pd.read_excel(os.getcwd()+'/120秒內解除轉轉樂紀錄.xlsx',usecols=["UserID","fireTime","timediff"])
print("reading Cheater file...")
df_cheater = pd.read_excel(os.getcwd()+'/cheaters.xlsx',usecols=["UserID"])
end_reading = perf_counter()
print("End of reading...")
print("-------------------------------------------------------------")
print(f"reading process cost Time : {format(end_reading-start_reading)}")
print("-------------------------------------------------------------")
#-------------------------------------------------------------------
# 將數據進行分類及提取出需要的訊息
def summary(summary:pd.DataFrame()):
    summary_all={}
    for i in summary.iterrows():
        temp = i[1].values
        userId = temp[0]
        date = datetime.datetime.strptime(temp[1],"%Y-%m-%d %H:%M:%S.%f")
        if userId not in summary_all.keys():
            summary_all[userId] = summary_all.get(userId,{})
        day_str="0"
        date_for_keys=0
        if date.day in range(1,10):
            day_str = day_str + str(date.day)
            date_for_keys = int(str(date.month)+day_str)
        else:
            date_for_keys = int(str(date.month)+str(date.day))
        if date_for_keys not in summary_all[userId].keys():
            summary_all[userId][date_for_keys] = summary_all.get(date_for_keys,{"First":date,"Last":date,"duration":"","times":0})
        
        # if summary_all[userId][date_for_keys]["Last"] < date:
        #     summary_all[userId][date_for_keys]["Last"] = date

        # summary_all[userId][date_for_keys]["duration"] = (summary_all[userId][date_for_keys]["Last"] - summary_all[userId][date_for_keys]["First"]).total_seconds()

        summary_all[userId][date_for_keys]["times"]+=1
    # 資料型態 : UserID:{day1:{"First":"","Last":"","duration":"",times:""},day2:{}}
    return summary_all
# 從資料中取得作弊者的資料
# def get_cheater_summary(summary:dict):
#     summary_cheater=dict()
#     for i in summary.keys():
#         for j in cheater.iterrows():
#             date = datetime.datetime.strptime(j["fireTime"],"%Y-%m-%d %H:%M:%S.%f")
#             if i in (cheater["UserID"].tolist()):
                
#     return summary_cheater

# 計算使用者的每個月平均登入時間
def cal_average_time(summary:dict,days:int):
    for i in df_Catch.iterrows():
        temp = i[1].values
        userId = temp[0]
        date = datetime.datetime.strptime(temp[1],"%Y-%m-%d %H:%M:%S.%f")
        day_str="0"
        date_for_keys=0
        if date.day in range(1,10):
            day_str = day_str + str(date.day)
            date_for_keys = int(str(date.month)+day_str)
        else:
            date_for_keys = int(str(date.month)+str(date.day))
        timediff = temp[2]
        if userId in summary.keys():
            if date_for_keys in summary[userId].keys():
                if summary[userId][date_for_keys]["First"] > date: # 日期較小 則替換
                    summary[userId][date_for_keys]["First"] = date
                if summary[userId][date_for_keys]["Last"] < date:  # 日期較大 則替換
                    summary[userId][date_for_keys]["Last"] = date
                if summary[userId][date_for_keys]["First"] != summary[userId][date_for_keys]["Last"]:   
                    summary[userId][date_for_keys]["duration"] = (summary[userId][date_for_keys]["Last"] - summary[userId][date_for_keys]["First"]).total_seconds()


    delkeys= {}
    for i in summary.keys():
        for j in summary[i].keys():
            if type(summary[i][j]["duration"]) == str:
                if i not in delkeys.keys():
                    delkeys[i] = delkeys.get(i,[])
                delkeys[i].append(j)
    for i in delkeys.keys():
        for j in delkeys[i]:
            summary[i].pop(j)

                    
       
    averageTime={}
    for i in summary.keys():
        Time = float(0.0)
        count = 0
        for j in summary[i].keys():
            Time += (summary[i][j]["duration"])
            count += (summary[i][j]["times"])
        averageTime[i]=[]
        if days == 0:
            averageTime[i].append(((Time/60)/60)/90)
            averageTime[i].append(count/90)
        else:
            averageTime[i].append(((Time/60)/60)/days)
            averageTime[i].append(count/days)
    return averageTime

# 計算轉轉樂平均標準差
def Standard_Deviation(summary:dict,block:int=1):
    # keys : UserID
    timediff_total = {}
    for i in df_Catch.iterrows():
        temp = i[1].values
        userId = temp[0]
        timediff = temp[2]
        date = datetime.datetime.strptime(temp[1],"%Y-%m-%d %H:%M:%S.%f")
        if userId in summary.keys():
            if userId not in timediff_total.keys():
                timediff_total[userId] = timediff_total.get(userId,[])
            else:
                timediff_total[userId].append(timediff)

    print("timediff_totoal",timediff_total)
    for i in timediff_total.keys():
        if i in summary.keys():
            if len(timediff_total[i]) == 0:
                summary[i].append(0)
            else:
                summary[i].append(sum(timediff_total[i])/len(timediff_total[i]))

            standard_deviation = np.std(timediff_total[i])
            if math.isnan(standard_deviation):
                summary[i].append(0)
            else:
                summary[i].append(standard_deviation)
    return summary

def FormatData(summary:dict):
    useless = {}
    userId = []
    average_play_time=[]
    average_sold_count=[]
    time_diff_middle=[]
    time_diff_standard_deviation=[]
    for key,values in summary.items():
        if len(values) == 4:
            userId.append(key)
            average_play_time.append(values[0])
            average_sold_count.append(values[1])
            time_diff_middle.append(values[2])
            time_diff_standard_deviation.append(values[3])
        else:
            useless[key] = useless.get(key,[])
            useless[key] = values

    print("useless",useless)
    data = {
        "userId":userId,
        "average_play_time":average_play_time,
        "average_sold_count":average_sold_count,
        "time_diff_middle":time_diff_middle,
        "time_diff_standard_deviation":time_diff_standard_deviation
    }
    df = pd.DataFrame(data)
    return df,useless

def get_cheater(summary:pd.DataFrame):
    check = []
    cheater = []
    for i in df_cheater.iterrows():
        temp = i[1].values
        userId = temp[0]
        cheater.append(userId)

    for i in summary["userId"]:
        if i in cheater:
            check.append(1)
        else:
            check.append(0)
    print("check length",len(check))
    summary.insert(loc=5,column="cheater",value=check)
    return summary

def KMans_data_analyze(summary:pd.DataFrame,n_clusters:int=2,title:str="Jan"):

    average_playTime_perMonth = list(summary["average_play_time"])
    standard_deviation = list(summary["time_diff_standard_deviation"])
    average_sold_count = list(summary["average_sold_count"])

    dataset = list(zip(average_playTime_perMonth,standard_deviation,average_sold_count))
    wcss = []
    for i in range(1,11):
        k_means = KMeans(n_clusters=i, random_state=42)
        k_means.fit(dataset)
        wcss.append(k_means.inertia_)
    plt.plot(np.arange(1,11),wcss)
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.show()

    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit(dataset)
    y = kmeans.predict(dataset)
    print("cluster",clusters)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(average_playTime_perMonth,standard_deviation, average_sold_count,c=y, cmap=plt.cm.Set1)
    plt.savefig(kmans_path+f"KMeans analyzer n_clusters={n_clusters}_3d.jpg")
    plt.show()
    # plt.xlabel("average_playTime_perMonth", fontweight = "bold")
    # plt.ylabel("standard_deviation",fontweight = "bold")
    # plt.title(f"KMeans analyzer n_clusters={n_clusters}_{title}",fontsize = 15, fontweight = "bold")
    # plt.scatter(average_playTime_perMonth, standard_deviation,average_sold_count, c=kmeans.labels_)
    # plt.savefig(kmans_path+f"KMeans analyzer n_clusters={n_clusters}.jpg")
    # plt.show()
    # plt.close() # 關閉圖表

def Decision_data_analyze(summary:pd.DataFrame):
    # 決策樹
    # 常見的決策亂度評估指標有 Information gain、Gain ratio、Gini index。
    #Parameters:
    # criterion: 亂度的評估標準，gini/entropy。預設為gini。
    # max_depth: 樹的最大深度。
    # splitter: 特徵劃分點選擇標準，best/random。預設為best。
    # random_state: 亂數種子，確保每次訓練結果都一樣，splitter=random 才有用。
    # min_samples_split: 至少有多少資料才能再分
    # min_samples_leaf: 分完至少有多少資料才能分
    y = summary["cheater"]
    X = summary.drop(['cheater','userId'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    print("X_train",X_train.shape,X_train)
    print("X_test",X_test.shape,X_test)
    print("y_train",y_train.shape,y_train)
    print("y_test",y_test.shape,y_test)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print("DecisionTreeClassifier model score",model.score(X_test, y_test))
    y_predict = model.predict(X_test)
    print("predicted",y_predict)
    print("accuracy",accuracy_score(y_test, y_predict))

    rank = []
    for i in range(1,6):
        model = DecisionTreeRegressor(max_depth=i)
        model.fit(X_train, y_train)
        # print("DecisionTreeRegressor model score",model.score(X_test, y_test))
        rank.append((i,model.score(X_test, y_test)))
    def takeSecond(elem):
        return elem[1]
    rank.sort(key=takeSecond)
    print("rank",rank)
    model = DecisionTreeRegressor(max_depth=rank[rank.__len__()-1][0])
    model.fit(X_train, y_train)
    print("DecisionTreeRegressor model score",model.score(X_test, y_test))
    y_predict = model.predict(X_test)
    y_predict = np.around(y_predict)
    # for i in y_predict:
    #     if y_predict[i] > 0.5:
    #         y_predict[i] = 1
    #     else:
    #         y_predict[i] = 0
    print("predicted",y_predict)
    print("accuracy",accuracy_score(y_test, y_predict))


def SVCR_data_analyze(summary:pd.DataFrame):
    # 參考 https://ithelp.ithome.com.tw/articles/10270447

    y = summary["cheater"]
    X = summary.drop(['cheater','userId'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    # 建立 linearSvc 模型
    linearSvcModel=svm.LinearSVC(C=1, max_iter=10000)
    # 使用訓練資料訓練模型
    linearSvcModel.fit(X_train, y_train)
    # 使用訓練資料預測分類
    predicted=linearSvcModel.predict(X_train)
    # 計算準確率
    accuracy = linearSvcModel.score(X_train, y_train)
    print("LinearSVC-----")
    print("predicted",predicted)
    print("accuracy",accuracy)

    # 建立 kernel='linear' 模型
    svcModel=svm.SVC(kernel='linear', C=1)
    # 使用訓練資料訓練模型
    svcModel.fit(X_train, y_train)
    # 使用訓練資料預測分類
    predicted=svcModel.predict(X_train)
    # 計算準確率
    accuracy = svcModel.score(X_train, y_train)
    print("kernel='linear'-----")
    print("predicted",predicted)
    print("accuracy",accuracy)

    polyModel=svm.SVC(kernel='poly', degree=3, gamma='auto', C=1)
    # 使用訓練資料訓練模型
    polyModel.fit(X_train, y_train)
    # 使用訓練資料預測分類
    predicted=polyModel.predict(X_train)
    # 計算準確率
    accuracy = polyModel.score(X_train, y_train)
    print("kernel='linear'-----")
    print("predicted",predicted)
    print("accuracy",accuracy)

    rbfModel=svm.SVC(kernel='rbf', gamma=0.7, C=1)
    # 使用訓練資料訓練模型
    rbfModel.fit(X_train, y_train)
    # 使用訓練資料預測分類
    predicted=rbfModel.predict(X_train)
    # 計算準確率
    accuracy = rbfModel.score(X_train, y_train)
    print("kernel='rbf'-----")
    print("predicted",predicted)
    print("accuracy",accuracy)
    # ------------------------------------------SVR
    # linearModel=svm.SVR(C=1, kernel='linear')
    # # 使用訓練資料訓練模型
    # linearModel.fit(X_train, y_train)
    # # 使用訓練資料預測分類
    # predicted=linearModel.predict(X_train)
    # accuracy = linearModel.score(X_train, y_train)
    # print("SVR kernel='linear'-----")
    # print("predicted",predicted)
    # print("accuracy",accuracy)


    # # 建立 kernel='poly' 模型
    # polyModel=svm.SVR(C=1e3, kernel='poly', degree=3, gamma='auto')
    # # 使用訓練資料訓練模型
    # polyModel.fit(X_train, y_train)
    # # 使用訓練資料預測分類
    # predicted=polyModel.predict(X_train)
    # accuracy = polyModel.score(X_train, y_train)
    # print("SVR kernel='poly'-----")
    # print("predicted",predicted)
    # print("accuracy",accuracy)

    # # 建立 kernel='rbf' 模型
    # rbfModel=svm.SVR(C=1e3, kernel='rbf', gamma='auto')
    # # 使用訓練資料訓練模型
    # rbfModel.fit(X_train, y_train)
    # # 使用訓練資料預測分類
    # predicted= rbfModel.predict(X_train)
    # accuracy = rbfModel.score(X_train, y_train)
    # print("SVR kernel='rbf'-----")
    # print("predicted",predicted)
    # print("accuracy",accuracy)
#-------------------------------------------------------------------
print("Starting process...")
start = perf_counter()

# 先將資料進行整合計算 datatype => UserID:{day1:{"First":"","Last":"","duration":"",times:""},day2:{}}
# summary_all_Jan = summary(df_Jan)
# summary_all_Feb = summary(df_Feb)
# summary_all_March = summary(df_March)

summary_all_Month = summary(pd.concat([df_Jan,df_Feb,df_March]))
print(summary_all_Month.items)
# 從轉轉樂取得名稱和登入時間
# summary_all_polygraph = get_cheater_summary(summary_all_Month)



# 在一個月內的平均登入時間計算 datatype => {UserID:averageTime}
print("Starting calculate averagetime...")
# average_time_Jan = cal_average_time(summary_all_Jan,31)# parameters => dict , int(per_month_days)
# average_time_Feb = cal_average_time(summary_all_Feb,28)
# average_time_March = cal_average_time(summary_all_March,31)

average_time_all = cal_average_time(summary_all_Month,0)
# print("average_time_all",average_time_all.items()) # {UserID:averageTime}


# 解除轉轉樂的平均標準差
print("Starting calculate Standard Deviation...")
# Jan_dataset = Standard_Deviation(average_time_Jan,1) # parameters => dict , int(month)
# Feb_dataset =Standard_Deviation(average_time_Feb,2)
# March_dataset =Standard_Deviation(average_time_March,3)
All_dataset = Standard_Deviation(average_time_all,0)
# print("All_dataset",All_dataset.items())
print(All_dataset.__len__()) # 100
# datatype : UserId:[Avg_playtime,sold_times,timediff_middle,timediff_standard_deviation]

# Format Data to pd.DataFrame
# col = [userId , average_play_time , average_sold_count , time_diff_standard_deviation]
All_data,useless = FormatData(All_dataset)
print(All_data.index)
# create cheater col
All_data = get_cheater(All_data)
print(All_data.head())

# KMeans clusters final output
print("Starting KMeans Cluster...")
# KMans_data_analyze(Jan_dataset,2,"Jan")
# KMans_data_analyze(Feb_dataset,2,"Feb") # parameters => dict , int(n_cluster)
# KMans_data_analyze(March_dataset,2,"March")
# 
#  
KMans_data_analyze(All_data,2,"All")
# KMans_data_analyze(All_data,3,"All")
# KMans_data_analyze(All_data,4,"All")
# KMans_data_analyze(All_data,5,"All")
# KMans_data_analyze(All_data,6,"All")

# Decision clusters final output
print("Starting Decision Tree Analyze...")
Decision_data_analyze(All_data)

print("Starting SVM Analyze...")
SVCR_data_analyze(All_data)

end = perf_counter()
print("End of processing...")
print("-------------------------------------------------------------")
print(f"reading process cost Time : {format(end_reading-start_reading)}")
print(f"processing time:{format(end-start)}")
print("-------------------------------------------------------------")
