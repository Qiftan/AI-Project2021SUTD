import pandas as pd
import numpy as np
import os
import pickle
import io
import re
from tqdm import tqdm
import utils
import math

def add_features(i):

    columns = utils.get_columns()

    X_train = pd.read_csv(datapath+"/covid_data_" + str(i) + ".csv", names=columns)
    # print(X_train.head()) 
    # print(X_train.info())

    # get time segment 0-23
    X_train["timeseg"] = X_train["timestamp"].str[11:13]
    X_train['timeseg'] = pd.to_numeric(X_train['timeseg'])
    X_train["date"] = X_train["timestamp"].str[-4:] + "-" + X_train["timestamp"].str[4:10]
    X_train["weekend"] = X_train["timestamp"].str[:3].isin(["Sun", "Sat"])
    X_train["weekend"] = X_train["weekend"].astype(int)
    # print('added time seg...')

    # exist or not features
    for col in ["entities", "hashtags", "mentions", "urls"]:
        X_train[col] = X_train[col].astype(str)
    X_train["entity_exist"] = X_train["entities"] != "null;"
    X_train["hashtag_exist"] = X_train["hashtags"] != "null;"
    X_train["mention_exist"] = X_train["mentions"] != "null;"
    X_train["url_exist"] = X_train["urls"] != "null;"
    # print('added exit or not features...')

    # h/e/m/url count
    X_train["entity_count"] = X_train["entities"].str.split(";").apply(
        lambda x: len([y for y in x if len(y) > 0 and y != "null"]))
    X_train["hashtag_count"] = X_train["hashtags"].str.split(" ").apply(
        lambda x: len([y for y in x if len(y) > 0 and y != "null;"]))
    X_train["mention_count"] = X_train["mentions"].str.split(" ").apply(
        lambda x: len([y for y in x if len(y) > 0 and y != "null;"]))
    X_train["url_count"] = X_train["urls"].str.split(":-: ").apply(
        lambda x: len([y for y in x if len(y) > 0 and y != "null;"]))
    # print('added count of h/e/m/url...')

    # approx length of tweets = sum of all h/e/m/url
    X_train["tlen"] = X_train["entity_count"] + X_train["hashtag_count"] + X_train["mention_count"] + X_train[
        "url_count"]

    X_train["entity_exist"] = X_train["entity_exist"].astype(int)
    X_train["hashtag_exist"] = X_train["hashtag_exist"].astype(int)
    X_train["mention_exist"] = X_train["mention_exist"].astype(int)
    X_train["url_exist"] = X_train["url_exist"].astype(int)
    # print("changed exit features to int type...")

    X_train = add_ratios(X_train) #*
    X_train = add_importance(X_train)
    X_train = add_year_month_date(X_train)
    X_train = add_day_of_week(X_train) #*
    X_train = add_sentiments(X_train) #*
    X_train = add_sine_cosine_hour(X_train)
    X_train = add_sine_cosine_day(X_train)
    X_train = add_sine_cosine_day_of_week(X_train)

    # print(X_train.head()) 
    # print(X_train.info())

    X_train.to_csv(datapath+"/covid_data_" + str(i) + ".csv")

def add_ratios(X_train):
    """#followers #friends #favorites"""
    X_train["#followers"] = X_train["#followers"].astype(float)
    X_train["ratio_fav_#followers"] = X_train["#favorites"] / (X_train["#followers"] + 1.0)
    X_train["ratio_fri_#followers"] = X_train["#friends"] / (X_train["#followers"] + 1.0)
    return X_train

def add_importance(X_train):
    """2019-SEP 30 will be zero then +1 """
    list_of_dates = list(X_train["date"].unique())
    tqdm.pandas()
    X_train["time_importance"] = X_train["date"].progress_apply(lambda x: list_of_dates.index(x)).values
    return X_train

def add_year_month_date(X_train):
    X_train["year"] = X_train["date"].str[:4]
    X_train["day"] = X_train["date"].str[-2:]
    m_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
              "Nov": 11, "Dec": 12}
    tqdm.pandas()
    X_train["month"] = X_train["date"].str[5:-3].progress_apply(lambda x: m_dict[x])
    X_train["year"] = X_train["year"].astype(int)
    X_train["day"] = X_train["day"].astype(int)
    return X_train

def add_day_of_week(X_train):
    m_dict = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    tqdm.pandas()
    X_train["day_of_week"] = X_train["timestamp"].str[:3].progress_apply(lambda x: m_dict[x])
    return X_train

def add_sentiments(X_train):
    # sentiments
    X_train['sentiment_p'], X_train['sentiment_n'] = X_train['sentiment'].str.split(' ', 1).str
    X_train['sentiment_p'] = pd.to_numeric(X_train['sentiment_p']) + 6.0  # to be positive for log scaling
    X_train['sentiment_n'] = pd.to_numeric(X_train['sentiment_n']) + 6.0
    # plus for training main, minus for patching
    X_train['sentiment_ppn'] = X_train['sentiment_p'] + X_train['sentiment_n']
    return X_train

def add_sine_cosine_hour(X_train):
    # hour
    X_train['sine_hour'] = np.sin(2*np.pi*(X_train["timeseg"].astype(float)/23))
    X_train['cosine_hour'] = np.cos(2*np.pi*(X_train["timeseg"].astype(float)/23))
    return X_train

def add_sine_cosine_day(X_train):
    # day
    X_train['sine_day'] = np.sin(2*np.pi*(X_train["day"].astype(float)/365))
    X_train['cosine_day'] = np.cos(2*np.pi*(X_train["day"].astype(float)/365))
    return X_train

def add_sine_cosine_day_of_week(X_train):
    # day
    X_train['sine_day_of_week'] = np.sin(2*np.pi*(X_train["day_of_week"].astype(float)/7))
    X_train['cosine_day_of_week'] = np.cos(2*np.pi*(X_train["day_of_week"].astype(float)/7))
    return X_train

def make_data(fromFilename, start=0):  
    MAXLINES = 10000
    csvfile = open(fromFilename, mode='r', encoding='utf-8')
    # or 'Latin-1' or 'CP-1252'
    filename = start
    for rownum, line in enumerate(csvfile):
        if rownum % MAXLINES == 0:
            filename += 1
            outfile = open(datapath +"/covid_data_" + str(filename) + '.csv', mode='w', encoding='utf-8')
        outfile.write(re.sub('\t',',',re.sub('(^|[\t])([^\t]*\,[^\t\n]*)', r'\1"\2"', line)))
        # outfile.write(line)
    outfile.close()
    csvfile.close()

datapath = 'data2_2'

if __name__ == "__main__":

    # make_data('raw_data/TweetsCOV19.tsv')
    # make_data('raw_data/TweetsCOV19_2.tsv')
    # make_data('raw_data/TweetsCOV19_3.tsv',1009)

    totalFile = 193
    for i in tqdm(range(1,totalFile+1)):
        if i != 153:
            add_features(i)

    # totalFile = 1009
    # for i in tqdm(range(817,totalFile+1)):
    #     add_features(i)

    # totalFile = 2013
    # for i in tqdm(range(1010,totalFile+1)):
    #     add_features(i)
