##############################################
### Program uses twitterscraper API        ###
### to scrap tweets between 2010 and 2020  ###
### containing ticker name or symbol       ###
### Output 1 csv file per ticker           ###
##############################################

import pandas as pd
import numpy as np
from twitterscraper import query_tweets
import datetime as dt
from datetime import datetime
import json


# Remove 5 companies: CAT, DIS, DOW, TRV, WBA
ticker = ["MMM OR 3M", "AXP OR American Express", "AAPL OR Apple", "BA OR Boeing", \
          "CVX OR Chevron", "CSCO OR Cisco", "KO OR Coca-Cola", "XOM OR Exxon Mobil", \
          "GS OR Goldman Sachs", "HD OR Home Depot", "IBM", "INTC OR Intel", \
          "JNJ OR Johnson & Johnson", "JPM OR JPMorgan Chase", "MCD OR McDonald's", \
          "MRK OR Merck", "MSFT OR Microsoft", "NKE OR Nike", "PFE OR Pfizer", \
          "PG OR Procter & Gamble", "UTX OR United Technologies", "UNH OR UnitedHealth", \
          "VZ OR Verizon", "V OR Visa", "WMT OR Wal-Mart"]

ticker_symbol = ["MMM", "AXP", "AAPL", "BA", \
                 "CVX", "CSCO", "KO", "XOM", \
                 "GS", "HD", "IBM", "INTC", \
                 "JNJ", "JPM", "MCD", \
                 "MRK", "MSFT", "NKE", "PFE", \
                 "PG",  "UTX", "UNH", 
                 "VZ", "V", "WMT"]

for i in range(len(ticker)):
    df = pd.DataFrame()
    for year in range(2010,2020):
        list_of_tweets = []
        num_days = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        for month in range(1,13):
            tweets = query_tweets(ticker[i], begindate=dt.date(year, month, 1), enddate=dt.date(year, month, num_days[month]), lang="en")
            list_of_tweets  += tweets
        print("Number of tweets:",len(list_of_tweets))
        
        list_of_json = []
        for tweets in list_of_tweets: # set date , change data type form Tweet to dict
            tweets.timestamp = datetime.strftime(tweets.timestamp, '%Y-%m-%d %H:%M:%S')
            list_of_json.append(vars(tweets))
        
        # Join year to form 10 years of tweets
        df = pd.concat([df, pd.DataFrame.from_records(list_of_json)])

    df.to_csv('../../Raw Data/Tweets/'+ticker_symbol[i]+'_tweets.csv', encoding='utf8')