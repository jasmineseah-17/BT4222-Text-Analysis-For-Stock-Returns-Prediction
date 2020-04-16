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

from pandas_datareader.data import DataReader

import nltk

nyse = mcal.get_calendar('NYSE')
nyse_holidays = nyse.holidays().holidays

### Initialize NLTK Variables ###
stop_words = stopwords.words("english")
punctuations = string.punctuation
wordnet_lemmatizer = WordNetLemmatizer()

def extract_text(path):
    '''
    Function to extract the relevant text information from 8K SEC .txt file
    Args:
      path: A string containing the directory of the file
    Returns:
      A tuple containing the relevant text information and the date of submission
    '''
    
    try: 
        with open(path) as f:
            soup = BeautifulSoup(f.read(), "lxml")

        submission_dt = soup.find("acceptance-datetime").string[:14]
        submission_dt = datetime.strptime(submission_dt, "%Y%m%d%H%M%S")

        # Extract HTML sections
        for section in soup.findAll("html"):
            try:
                section = unicodedata.normalize("NFKD", section.text)
                section = section.replace("\t", " ").replace("\n", " ").replace("/s", " ").replace("\'", "'")
            except AttributeError:
                section = str(section.encode('utf-8'))
        filing = "".join((section))
        
        filing = html2text.html2text(filing)
        filing = filing.replace("\t", " ").replace("\n", " ").replace("/s", " ").replace("\'", "'").replace("/", " ")
    
    except:
        with open(path) as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        submission_dt = soup.find("acceptance-datetime").string[:14]
        submission_dt = datetime.strptime(submission_dt, "%Y%m%d%H%M%S")

        # Extract HTML sections
        for section in soup.findAll("html"):
            try:
                section = unicodedata.normalize("NFKD", section.text)
                section = section.replace("\t", " ").replace("\n", " ").replace("/s", " ").replace("\'", "'")
            except AttributeError:
                section = str(section.encode('utf-8'))
        filing = "".join((section))
        
        filing = html2text.html2text(filing)
        filing = filing.replace("\t", " ").replace("\n", " ").replace("/s", " ").replace("\'", "'").replace("/", " ")
    


    return filing, submission_dt

def extract_items(path):
    '''
    Function to extract the event-item from 8K SEC .csv file
    Args:
      path: A string containing the directory of the file
    Returns:
      A list of the relevant event-items. 
    '''
    
    try: 
        with open(path) as f:
            soup = BeautifulSoup(f.read(), "lxml")
        
        pattern = re.compile("ITEM INFORMATION[^\n]*\n")
        item_list = re.findall(pattern, soup.get_text())
        item_list = [item.replace("\t", " ").replace("\n", "") for item in item_list]
        
    except:
        with open(path) as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        pattern = re.compile("ITEM INFORMATION[^\n]*\n")
        item_list = re.findall(pattern, soup.get_text())
        item_list = [item.replace("\t", " ").replace("\n", "") for item in item_list]
        

    return item_list

def parallelize(data, func, num_of_processes=8):
    '''
    Run the function on multi-processing/ in parallel fashion.
    It does so by splitting the dataframe into n number of parts where n is the number of parallel processes.
    The function is the applied concurrently on these n number of parts.
    
    Args:
      data: A Pandas `dataframe`
      func: Function to be carried out.
      num_of_processes: Number of parallel cores to use.
    
    Returns:
      data: A Pandas `dataframe` after func is performed
    '''
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    '''
    Apply function on subset of dataframe.
    
    Args:
      func: Function to be carried out.
      data_subset: A Pandas `dataframe`
    
    Returns:
      data_subset: A Pandas `dataframe` after func is performed
    '''
    return data_subset.apply(func, axis=1)

def parallel_extract_text_items(df):
    '''
    Apply extract_text in parallel fashion. Two new columns will be created: 
    (1) text: This contains the relevant text information from the scraped 8K SEC file
    (2) items: This contains the associated list of even-items
    Args:
      df: A Pandas `dataframe`
    Returns:
      df: A Pandas `dataframe` after extract_text is performed
    '''
    df['text'], df['release_date'] = zip(*df['directory'].apply(extract_text))
    df['items'] = df["directory"].map(extract_items)
    return df

def find_between(s, first, last=None):
    '''
    Helper function to find the text that is between a start word and end word if specified.
    Args:
      s: A string; text
      first: A string; first word
      last: A string; last word
    Returns:
      A string of text between the start and end word if specified.
    '''
    try:
        if last != None:
            start = [m.start() for m in re.finditer(first, s)][-1]
            start = start + len(first)
            end = [m.start() for m in re.finditer(last, s)][-1]
            
            return s[start:end]
        else:
            start = [m.start() for m in re.finditer(first, s)][-1]
            start = start + len(first)
            
            return s[start:]
    except IndexError:
        return ""


def preprocess_text_lemm(text):
    '''
    Normalised text by lemmatizing. 
    Args:
      text: A string of words.
    Return:
      text: A string of normalised words.
    '''
    words = set(nltk.corpus.words.words())
    text = text.replace("\x90", " ").replace("\x91", " ").replace("\x92", " ").replace("\x93", " ").replace("\x94", " ").replace("\x95", " ").replace("\x96", " ").replace("\x97", " ").replace("\x98", " ").replace("\x99", " ")
    self_generated_stopwords = ["and", "the", "for", "with", "note", "with", "date", "this", "any", "it", "such", "will", "that", "shall", "are", "be", "have", "its", "not", "under", "which"]
    try:
        # Generate tokens
        tokens = word_tokenize(text)

        # Remove unwanted characters, numbers and symbols
        tokens = [t for t in tokens if t.isalpha()]
        tokens = list(filter(lambda t: t not in punctuations, tokens))
        filtered_tokens = []
        
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        # Convert to lowercase
        filtered_tokens = [t.lower() for t in filtered_tokens]

        ## Remove short words (< 3 letters)
        filtered_tokens = list(filter(lambda t: len(t) > 2, filtered_tokens))

        # Lemmatize (including the verbs!)
        filtered_tokens = list(map(lambda token: wordnet_lemmatizer.lemmatize(token.lower(), pos='v'), filtered_tokens))
        
        # Remove stopwords [self-generated]
        filtered_tokens = list(filter(lambda t: t not in self_generated_stopwords, filtered_tokens))
        
        # Remove gibberish
        filtered_tokens = list(filter(lambda t: t in words, filtered_tokens))

        return filtered_tokens
    
    except Exception as e:
        raise e

def preprocess_text_stem(text):
    '''
    Normalised text by stemming
    Args:
      text: A string of words
    Return:
      text: A string of normalised words. 
    '''
    words = set(nltk.corpus.words.words())
    text = text.replace("\x90", " ").replace("\x91", " ").replace("\x92", " ").replace("\x93", " ").replace("\x94", " ").replace("\x95", " ").replace("\x96", " ").replace("\x97", " ").replace("\x98", " ").replace("\x99", " ")
    self_generated_stopwords = ["and", "the", "for", "with", "note", "with", "date", "this", "any", "it", "such", "will", "that", "shall", "are", "be", "have", "its", "not", "under", "which"]
    try:
        # Generate tokens
        tokens = word_tokenize(text)

        # Remove unwanted characters, numbers and symbols
        tokens = [t for t in tokens if t.isalpha()]
        tokens = list(filter(lambda t: t not in punctuations, tokens))
        filtered_tokens = []
        
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        # Convert to lowercase
        filtered_tokens = [t.lower() for t in filtered_tokens]

        ## Remove short words (< 3 letters)
        filtered_tokens = list(filter(lambda t: len(t) > 2, filtered_tokens))
        
        # Remove gibberish
        filtered_tokens = list(filter(lambda t: t in words, filtered_tokens))
        
        # Remove stopwords [self-generated]
        filtered_tokens = list(filter(lambda t: t not in self_generated_stopwords, filtered_tokens))
        
        # Stemm (including the verbs!)
        #filtered_tokens = list(map(lambda token: wordnet_lemmatizer.lemmatize(token.lower(), pos='v'), filtered_tokens))
        filtered_tokens = list(map(lambda token: SnowballStemmer("english").stem(token.lower()), filtered_tokens))
        

        return filtered_tokens
    
    except Exception as e:
        raise e

def get_tokens_stem(row):
    '''
    Function to get tokens and remove white space from stemmed words.
    Args:
      row: A pandas dataframe row
    Returns:
      List of tokens
    '''
    result = []

    items = row["items"]
    exhibits = row["exhibits"]
    text = row["text"]
    
    for i in range(len(items)):
        if i == len(items) - 1:
            if len(exhibits) != 0:
                sub_text = find_between(text, items[i], exhibits[0])
            else:
                sub_text = find_between(text, items[i])
        else:
            sub_text = find_between(text, items[i], items[i+1])

        sub_tokens = preprocess_text_stem(sub_text)

        result += sub_tokens

    for i in range(len(exhibits)):
        if i == len(exhibits) - 1:
            sub_text = find_between(text, exhibits[i])
        else:
            sub_text = find_between(text, exhibits[i], exhibits[i+1])

        new_text = " ".join(sub_text.split(" ")[:200])
        sub_tokens = preprocess_text_stem(new_text)

        result += sub_tokens
    
    return result
    
def get_tokens_lemm(row):
    '''
    Function to get tokens and remove white space from lemmatized words.
    Args:
      row: A pandas dataframe row
    Returns:
      List of tokens
    '''
    result = []

    items = row["items"]
    exhibits = row["exhibits"]
    text = row["text"]
    
    for i in range(len(items)):
        if i == len(items) - 1:
            if len(exhibits) != 0:
                sub_text = find_between(text, items[i], exhibits[0])
            else:
                sub_text = find_between(text, items[i])
        else:
            sub_text = find_between(text, items[i], items[i+1])

        sub_tokens = preprocess_text_lemm(sub_text)

        result += sub_tokens

    for i in range(len(exhibits)):
        if i == len(exhibits) - 1:
            sub_text = find_between(text, exhibits[i])
        else:
            sub_text = find_between(text, exhibits[i], exhibits[i+1])

        new_text = " ".join(sub_text.split(" ")[:200])
        sub_tokens = preprocess_text_lemm(new_text)

        result += sub_tokens
    
    return result

def parallel_process_text_stem(df):
    '''
    Apply get_tokens_stem in parallel fashion. One new column will be created: 
    (1) processed_text_stem: This contains stemmed words
    Args:
      df: A Pandas `dataframe`
    Returns:
      df: A Pandas `dataframe` after get_tokens_stem is performed
    '''
    
    df["processed_text_stem"] = df.apply(get_tokens_stem, axis=1)
    return df

def parallel_process_text_lemm(df):
    '''
    Apply get_tokens_lemm in parallel fashion. One new column will be created: 
    (1) processed_text_lemm: This contains lemmatized words
    Args:
      df: A Pandas `dataframe`
    Returns:
      df: A Pandas `dataframe` after get_tokens_stem is performed
    '''
    
    df["processed_text_lemm"] = df.apply(get_tokens_lemm, axis=1)
    return df

def get_data_for_multiple_stocks(tickers, start_date, end_date):
    '''
    Obtain stocks information (Date, OHLC, Volume and Adjusted Close). 
    Uses Pandas DataReader to make an API Call to Yahoo Finance and download the data directly.
    Computes other values - Log Return and Arithmetic Return.
    
    Args: 
      tickers: List of Stock Tickers
      start_date: Start Date of the stock data
      end_date: End Date of the stock data
    Returns:
      A dictionary of dataframes for each stock
    '''
    stocks = dict()
    for ticker in tickers:
        s = DataReader(ticker, 'yahoo', start_date, end_date)
        s.insert(0, "Ticker", ticker)  #insert ticker column so you can reference better later
        s['Date'] = pd.to_datetime(s.index) #useful for transformation later
        s['Prev Adj Close'] = s['Adj Close'].shift(1)
        s['Log Return'] = np.log(s['Adj Close']/s['Prev Adj Close'])
        s['Return'] = (s['Adj Close']/s['Prev Adj Close']-1)
        s = s.reset_index(drop=True)
        
        cols = list(s.columns.values) # re-arrange columns
        cols.remove("Date")
        s = s[["Date"] + cols]
        s["Date"] = pd.to_datetime(s["Date"])
        s = s.set_index("Date")
        
        stocks[ticker] = s
        
    return stocks

def get_price(ticker,start_date,end_date, stock_data):
    '''
    Function to get the price of the ticker based on the start and end date
    Args:
      ticker: A string; stock ticker
      start_date: Start Date
      end_date: End Date
      stock_data: A Pandas dataframe of the OHLC and volume of the stock
    Returns:
      Price of the ticker.
    '''
    start_date = start_date.date()
    end_date = end_date.date()
    
    try: 
        price = stock_data[ticker].loc[start_date:end_date,"Adj Close"].mean()
    except:
        price = np.nan
    
    return price

def weekday_check_start_date(date):
    '''
    Function to check whether the start_date is a trading day. If it is not a trading day, 
    function will take the previous trading day. 
    Args:
      date: Date
    Returns:
      A trading date.
    '''
    while date.isoweekday() > 5 or date.date() in nyse_holidays:
        date = date + timedelta(days=-1)
    return date

def weekday_check_end_date(date):
    '''
    Function to check whether the end_date is a trading day. If it is not a trading day, 
    function will take the next trading day. 
    Args:
      date: Date
    Returns:
      A trading date.
    '''
    while date.isoweekday() > 5 or date.date() in nyse_holidays:
        date = date + timedelta(days=1)
    
    return date

def calculate_rtns(end_value,start_value):
    '''
    Function to calculate log returns. 
    Args:
      end_value: A float; price
      start_value: A float; price
    Returns:
      Log returns (float)
    '''
    rtns = np.log(end_value/start_value)
    return rtns

def get_start_end_dates(row, lag_period=5):
    '''
    Function to get the start_date and end_date based on the release_date. In this case,
    the start_date is the release_date and the end_date is lag_period after the release_date.
    Args:
      row: A Pandas dataframe row
      lag_period: An integer; The number of days after the release_date
    Returns:
      A tuple containing the final start_date and end_date after adjusting for trading dates.
    '''
    release_date = row["date"]
    
    start_date = weekday_check_start_date(release_date)
    end_date = weekday_check_end_date(release_date + timedelta(days=lag_period))
    
    return (start_date, end_date)

def get_start_end_prices(row, start_date, end_date, ticker, stock_data):
    '''
    Function to get the start and end prices based on the start_date and end_date
    Args:
      row: A Pandas dataframe row
      start_date: Start Date
      end_date: End Date
      ticker: A string; stock ticker
      stock_data: A Pandas dataframe of the OHLC and volume of the stock
    Returns:
      A tuple containing the price before the release date and price after the release date
    '''
    start_date = row[start_date]
    end_date = row[end_date]
    ticker = row[ticker]
    
    price_before_release = get_price(ticker, start_date, start_date, stock_data)
    price_after_release = get_price(ticker, end_date, end_date, stock_data)
    
    return (price_before_release, price_after_release)
















