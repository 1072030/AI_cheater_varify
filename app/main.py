import os
import sys
import pandas as pd
import numpy as np
import nltk 
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
# dir_path = os.path.dirname(os.path.realpath(__file__)) C:\Users\ASUS\Desktop\code\cheater_analyze\app
# os.getcwd() C:\Users\ASUS\Desktop\code\cheater_analyze
# sheet_name list ["一億以上亂幣交易紀錄","賣商店次數前100名角色","賣商店次數前100名角色1月賣商店紀錄","賣商店次數前100名角色2月賣商店紀錄",
#                  "賣商店次數前100名角色3月賣商店紀錄","120秒內解除轉轉樂紀錄","轉轉樂啟動次數記錄"]
# --------------------------------------------------------------
# key data needed  usecols=[]
# 賣商店次數前100名角色             : ["UserID","SoldCount"]
# 賣商店次數前100名角色1月賣商店紀錄 : ["UserID","ExchangeDate"]
# 120秒內解除轉轉樂紀錄             : ["UserID","timediff"]
# 轉轉樂啟動次數記錄                : ["UserID","Times"]
# --------------------------------------------------------------
# What are we search for ?
# Y = f(x)
# Y 可以是 boolean 是否為外掛
# Y 也可以是 此使用者是外掛的機率
# 難度差異蠻大
# x 玩家的一天的登入時間，解除轉轉樂時間和一天的次數
# --------------------------------------------------------------
# print(sys.argv)
print("reading Jan file...")
df_Jan = pd.read_excel(os.getcwd()+'/賣商店次數前100名角色1月賣商店紀錄.xlsx',usecols=["UserID","ExchangeDate"])
# print("reading Feb file...")
# df_Feb = pd.read_excel(os.getcwd()+'/賣商店次數前100名角色2月賣商店紀錄.xlsx',usecols=["UserID","ExchangeDate"])
# print("reading March file...")
# df_March = pd.read_excel(os.getcwd()+'/賣商店次數前100名角色3月賣商店紀錄.xlsx',usecols=["UserID","ExchangeDate"])
print("reading Catching AI file")
df_Catch = pd.read_excel(os.getcwd()+'/120秒內解除轉轉樂紀錄.xlsx',usecols=["UserID","fireTime","timediff"])
print("reading Cheater file...")
cheater = pd.read_excel(os.getcwd()+'/cheaters.xlsx',usecols=["UserID"])
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
        if date.day not in summary_all[userId].keys():
            summary_all[userId][date.day] = summary_all.get(date.day,{"First":date,"Last":date,"duration":"","times":0})
        
        if summary_all[userId][date.day]["Last"] < date:
            summary_all[userId][date.day]["Last"] = date
        summary_all[userId][date.day]["Last"] = date
        summary_all[userId][date.day]["duration"] = (summary_all[userId][date.day]["Last"] - summary_all[userId][date.day]["First"]).total_seconds()
        summary_all[userId][date.day]["times"]+=1
    # 資料型態 : UserID:{day1:{"First":"","Last":"","duration":"",times:""},day2:{}}
    return summary_all
# 從資料中取得作弊者的資料
def get_cheater_summary(summary_all:dict):
    summary_cheater=dict()
    for i in summary_all.keys():
        if i in (cheater["UserID"].tolist()):
            summary_cheater[i]=summary_all[i]
    return summary_cheater

def cal_average_time(summary:dict,days:int):
    averageTime=[]
    for i in summary.keys():
        Time=0.0
        for j in summary[i].keys():
            Time += summary[i][j]["duration"]
        averageTime.append(((Time/60)/60)/days)
    return averageTime
            

    
#-------------------------------------------------------------------
print("Starting process...")
summary_all_Jan = summary(df_Jan)
# summary_all_Feb = summary(df_Feb)
# summary_all_March = summary(df_March)

# summary_cheater_Jan = get_cheater_summary(summary_all_Jan)
# summary_cheater_Feb = get_cheater_summary(summary_all_Feb)
# summary_cheater_March = get_cheater_summary(summary_all_March)

print("Starting calculate averagetime...")
average_time = cal_average_time(summary_all_Jan,31)
print(average_time)
# 在一個月內的平均登入時間計算
# 在
