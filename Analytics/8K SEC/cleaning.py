import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import string
import re
import unicodedata
import os
import multiprocessing
import math
import nltk
import html2text
import pandas_market_calendars as mcal

from tqdm import tqdm
from time import sleep

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from time import sleep
from datetime import datetime, timedelta
from os.path import isdir, join

from multiprocessing import  Pool
from functools import partial

from cleaning_helper import *

START_DATE = "2010-01-01"
END_DATE = "2019-12-31"
time_horizons = [1, 3, 5, 10, 15, 20, 25, 30]

### Map Exhibit and Items to Specific Events of the 8K SEC Filings ###
registrant_biz = ["Entry into a Material Definitive Agreement",
                  "Termination of a Material Definitive Agreement",
                  "Bankruptcy or Receivership",
                  "Mine Safety - Reporting of Shutdowns and Patterns of Violations"
                 ]

financial_info = ["Completion of Acquisition or Disposition of Assets",
                  "Results of Operations and Financial Condition",
                  "Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement of a Registrant",
                  "Triggering Events That Accelerate or Increase a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement",
                  "Costs Associated with Exit or Disposal Activities",
                  "Material Impairments"
                 ]

sec_trading = ["Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing",
               "Unregistered Sales of Equity Securities",
               "Material Modification to Rights of Security Holders"
              ]

acc_financial = ["Changes in Registrant's Certifying Accountant",
                 "Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim Review"
                ]

corp_gov = ["Changes in Control of Registrant",
            "Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; Compensatory Arrangements of Certain Officers",
            "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
            "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
            "Amendment to Registrant's Code of Ethics, or Waiver of a Provision of the Code of Ethics",
            "Change in Shell Company Status",
            "Submission of Matters to a Vote of Security Holders",
            "Shareholder Director Nominations"
           ]

asset_sec = ["ABS Informational and Computational Material",
             "Change of Servicer or Trustee",
             "Change in Credit Enhancement or Other External Support",
             "Failure to Make a Required Distribution",
             "Securities Act Updating Disclosure"
            ]

regulation_fd = ["Regulation FD Disclosure"]

other_events = ["Other Events (The registrant can use this Item to report events that are not specifically called for by Form 8-K, that the registrant considers to be of importance to security holders.)",  "Other Events"]

financial_exhibits = ["Financial Statements and Exhibits"]

events_mapper = {
    "Registrant's Business and Operations": registrant_biz,
    "Financial Information": financial_info,
    "Securities and Trading Markets": sec_trading,
    "Matters Related to Accountants and Financial Statements": acc_financial,
    "Corporate Governance and Management": corp_gov,
    "Asset-Backed Securities": asset_sec,
    "Regulation FD": regulation_fd,
    "Other Events": other_events,
    "Financial Statements and Exhibits": financial_exhibits
}

all_events = list(events_mapper.keys())


### Load CIK Mapper and Relevant Stock Data ###
cik_df = pd.read_csv("cik_mapper.csv")
cik_df = cik_df.drop("Unnamed: 0", axis=1)
cik_df["CIK"] = cik_df["CIK"].map(lambda x:str(x))

stock_data = get_data_for_multiple_stocks(cik_df.Symbol.values.tolist(), START_DATE, END_DATE)


### PATH Information ###
path = "../../Raw Data/8K SEC"
ciks = [f for f in os.listdir(path) if isdir(join(path, f))]
path_dict = {}
for cik in ciks:
    path_cik = path + "/" + cik
    txt_files = os.listdir(path_cik)
    lst = []
    for txt_file in txt_files:
        path_txt = path_cik + "/" + txt_file
        lst.append(path_txt)
    
    path_dict[cik] = lst

events_path = "../../Raw Data/8K SEC Types"
path_dict_events = {}
for cik in ciks:
    path_cik = events_path + "/" + cik
    txt_files = os.listdir(path_cik)
    lst = []
    for txt_file in txt_files:
        path_txt = path_cik + "/" + txt_file
        lst.append(path_txt)

    path_dict_events[cik] = lst

for i in tqdm(range(len(ciks))):
    
    ### 1. Extract Text ###
    cik = ciks[i]
    print(cik)
    df = pd.DataFrame(columns = ["cik", "directory", "date"])
    print(df.head())
    for path in path_dict[cik]:
        df = df.append({
            "cik": cik, 
            "directory": path,
            "date": path.split("/")[-1].split("_")[-1].split(".")[0].split("_")[0] 
            }, ignore_index=True)

    results = parallelize(df, parallel_extract_text_items, multiprocessing.cpu_count())
    print(results.head())

    ### 2. Extract Events and Exhibits ###
    exhibit_df = pd.DataFrame(columns=["cik", "Description", "Type", "Date"])
    for path in path_dict_events[cik]:
        sub_df = pd.read_csv(path)
        sub_df["cik"] = cik
        sub_df["Date"] = path.split("/")[-1].split("_")[-1].split(".")[0].split("_")[0]
        sub_df = sub_df[["cik", "Description", "Type", "Date"]]
        exhibit_df = pd.concat([exhibit_df, sub_df])

    print(exhibit_df.head())
    results["descriptions"], results["exhibits"] = None, None
    for index, row in results.iterrows():
        sub_df = exhibit_df[exhibit_df["Date"] == row["date"]]
        descriptions = sub_df["Description"].tolist()
        types = sub_df["Type"].tolist()
        
        filtered_descriptions = list(filter(lambda x: x not in ["FORM 8-K", 
                                                                "Complete submission text file", 
                                                                np.nan, "GRAPHIC", "8-K"], descriptions))
        filtered_types = list(filter(lambda x: x not in ["8-K", np.nan, "GRAPHIC"], types))
        
        results.at[index, "descriptions"] = filtered_descriptions
        results.at[index, "exhibits"] = filtered_types

    print(results.head())
    results["items"] = results["items"].map(lambda x:[i.split("ITEM INFORMATION:  ")[-1] for i in x])
    
    for event in all_events:
        results[event] = 0

    for index, row in results.iterrows():
        items = row["items"]
        for item in items:
            item = item.replace(":", ";")
            for key in events_mapper:
                if item in events_mapper[key]:
                    main_event = key
                    break
            results.loc[index, main_event] = 1

    ### 3. Text Cleaning (Stemming and Lemmatization) ###
    results = parallelize(results, parallel_process_text_stem, multiprocessing.cpu_count())
    results = parallelize(results, parallel_process_text_lemm, multiprocessing.cpu_count())

    ### 4. Combine Texts on the Same Date ###
    results = results.sort_values("release_date")
    results = results.reset_index().drop("index", axis=1)

    results.date = pd.to_datetime(results.release_date.dt.date)
    stem_combined_text_df = results[["date", "processed_text_stem"]].groupby(["date"], as_index=False).agg(sum)
    lemm_combined_text_df = results[["date", "processed_text_lemm"]].groupby(["date"], as_index=False).agg(sum)

    intermediate_df = results[[i for i in results.columns if i not in ["processed_text_stem", "processed_text_lemm", "release_date"]]]
    intermediate_df = intermediate_df.groupby(["date"]).first().reset_index()
    results = stem_combined_text_df.merge(intermediate_df, how="left", on="date")
    results = lemm_combined_text_df.merge(results, how="left", on="date")

    ### 5. Get Price Data ###
    ticker = cik_df[cik_df.CIK == cik]["Symbol"].values[0]
    sector = cik_df[cik_df.CIK == cik]["GICS Sector"].values[0]
    security = cik_df[cik_df.CIK == cik]["Security"].values[0]
    
    results["ticker"] = ticker
    results["sector"] = sector
    results["security"] = security
    

    for i in time_horizons:
        results['start_date'], results["end_date_{}".format(i)] = zip(*results.apply(get_start_end_dates, lag_period=i, axis=1))
        results['start_price'], results["end_price_{}".format(i)] = zip(*results.apply(get_start_end_prices,
                                                                                       start_date = "start_date",
                                                                                       end_date = "end_date_{}".format(i),
                                                                                       ticker = "ticker",
                                                                                       stock_data = stock_data,
                                                                                       axis=1))
        results["rtns_{}".format(i)] = results.apply(lambda row: calculate_rtns(row["end_price_{}".format(i)],
                                                                                row["start_price"]
                                                                               ), axis=1)
        results["signal_{}".format(i)] = results["rtns_{}".format(i)].map(lambda x: "up" if x > 0 else "down")
    


    print(results.head())
    save_path = "../../Processed Data/8K SEC/"
    results.to_pickle(save_path + cik + "_df.pkl")
    






