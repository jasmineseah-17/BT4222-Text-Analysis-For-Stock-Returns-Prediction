#########################################################################
### Program clean tweets                                              ###
### 1. spaCy POS tagging for relevant tweets (apple fruit vs iphone)  ###
### 2. Sentiment analysis of tweets                                   ###
### 3. Group tweets by date                                           ###
### 4. Process tweets by removing URLs, hashtags, emoticons           ###
### 5. Feature engineering                                            ###
### 6. Tokenise, remove stopwords, lemmatise tweets                   ###
### 7. Join with prices, derive price features and target label       ###
### Output 1 pickle per ticker                                        ###
#########################################################################

""" Copyright 2017, Dimitrios Effrosynidis, All rights reserved. """
## Credit for NLP cleaning portion

import pandas as pd
import numpy as np
import json

import string
import ast
from datetime import timedelta

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
stoplist = stopwords.words('english')
my_stopwords = "multiExclamation multiQuestion multiStop url atUser st rd nd th am pm" # my extra stopwords
stoplist = stoplist + my_stopwords.split()
lemmatizer = WordNetLemmatizer() # set lemmatizer

from techniques import *

import spacy
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

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

########################################################################
### 1. spaCy POS tagging for relevant tweets (apple fruit vs iphone) ###
########################################################################
def spacy_pos(df, name):
    '''
    POS-tag each token and filter for texts with "ORG" label
    
    Parameters
    ----------
        df         (pandas DataFrame) 
        name       (string) ticker name
        
    Returns
    -------
        the processed pandas DataFrame
    '''
    def find_org(text, name):
        doc = nlp(text)
        for ent in doc.ents:
#             print(ent.text, ent.label_)
            if (ent.text.lower()==name.lower()) & (ent.label_=='ORG'):
                return True
        return False
    
    df['relevant'] = [find_org(text,name) for text in df['text']]
    print("Before:", df.shape)
    df = df[(df['relevant']==True)]
    print("After:", df.shape)
    return df

########################################################################
### 2. Sentiment analysis of tweets                                  ###
### 3. Group tweets by date                                          ###
########################################################################
def group_tweets_by_date(df, symbol, name):
    '''
    Aggregate all columns after grouping rows by dates. 
    Shift weekend tweets to following Monday.
    
    Parameters
    ----------
        df         (pandas DataFrame)
        symbol     (string) ticker symbol eg. AAPL
        name       (string) ticker name eg. Apple
        
    Returns
    -------
        the processed pandas DataFrame
    '''
    df_filter = df[["text", "hashtags", "likes", "replies", "parent_tweet_id", "timestamp"]]
    df_filter.likes = df.likes.astype('int64')
    df_filter.replies = df.replies.astype('int64')

    # remove retweets
    df_filter = df_filter[df_filter.parent_tweet_id.isnull()]
    
    df_filter['hashtags'] = df_filter['hashtags'].apply(ast.literal_eval)
    df_filter['hashtags'] = df_filter['hashtags'].apply(lambda x : ','.join(x))
    df_filter['timestamp'] = pd.to_datetime(df_filter['timestamp'])
    df_filter['day'] = df_filter['timestamp'].dt.dayofweek
    df_filter['vader'] = [analyser.polarity_scores(tweet)['compound'] for tweet in df_filter['text']]
    
    # carry forward weekend tweets to following Monday (1 or 2 days)
    df_filter['stock_date'] = np.where(df_filter['day']>4,
                                      df_filter['timestamp'] + pd.to_timedelta(7-df_filter['day'], unit='d'), 
                                      df_filter['timestamp']
                                     )
    # group tweets by dates
    df_filter['stock_date'] = df_filter['stock_date'].dt.date
    df_filter = df_filter.groupby(df_filter['stock_date']).agg({'text': lambda x: ','.join(x), 
                                                                'hashtags': lambda x: ','.join(x), 
                                                                'likes':'sum', 
                                                                'replies': 'sum',
                                                                'vader': 'mean'
                                                               })
    df_filter['hashtags'] = df_filter['hashtags'].apply(lambda hashtags: list(filter(None, hashtags.split(','))))

    df_filter['text_removeCompany'] = df_filter.text.str.replace(symbol+' ','')
    name = name.lower()
    df_filter['text_removeCompany'] = df_filter.text_removeCompany.str.lower().str.replace(name+" ",'')
    df_filter = df_filter.reset_index(drop=False)

    return df_filter

########################################################################
### 6. Tokenise, remove stopwords, lemmatise tweets                  ###
########################################################################
def tokenize(text):
    '''
    Tokenise texts, remove stopwords, lemmatise word.
    
    Parameters
    ----------
        text         (string)
        
    Returns
    -------
        list of tokens (string)
    '''
    onlyOneSentenceTokens = [] # tokens of one sentence each time
    
    tokens = word_tokenize(text)
    tokens = replaceNegations(tokens) 
    translator = str.maketrans('', '', string.punctuation) 
    text = text.translate(translator) # Remove punctuation

    tokens = nltk.word_tokenize(text)

    for w in tokens:
        if (w not in stoplist):
            final_word = w.lower() 
            final_word = replaceElongated(final_word)
            final_word = lemmatizer.lemmatize(final_word)
            onlyOneSentenceTokens.append(final_word)           

    onlyOneSentence = " ".join(onlyOneSentenceTokens) # form again the sentence from the list of tokens
        
    return onlyOneSentenceTokens

########################################################################
### 4. Process tweets by removing URLs, hashtags, emoticons          ###
### 5. Feature engineering of numerical features                     ###
########################################################################
# A clean tweet should not contain URLs, hashtags (i.e. #happy) or mentions (i.e. @BarackObama)
def clean_dirty_tweets(text_series):
    '''
    Clean tweets before tokenisation.
    
    Parameters
    ----------
        text_series    (pandas Series)
        
    Returns
    -------
        the pandas DataFrame containing processed text 
        and other engineered features
    '''
    clean_tweets = []
    
    for text in text_series:
        totalEmoticons = 0
        totalSlangs = 0
        totalSlangsFound = []
        totalElongated = 0
        totalMultiExclamationMarks = 0
        totalMultiQuestionMarks = 0
        totalMultiStopMarks = 0
        totalAllCaps = 0

        text = removeUnicode(text)
        text = replaceURL(text)
        text = replaceAtUser(text)
        text = removeWholeHashtag(text)

        temp_slangs, temp_slangsFound = countSlang(text)
        totalSlangs += temp_slangs
        for word in temp_slangsFound:
            totalSlangsFound.append(word) # all the slangs found in all sentences

        text = replaceSlang(text)
        text = replaceContraction(text)
        text = removeNumbers(text)

        emoticons = countEmoticons(text) 
        totalEmoticons += emoticons

        text = removeEmoticons(text)
        totalAllCaps += countAllCaps(text)

        totalMultiExclamationMarks += countMultiExclamationMarks(text)
        totalMultiQuestionMarks += countMultiQuestionMarks(text) 
        totalMultiStopMarks += countMultiStopMarks(text)
        text = replaceMultiExclamationMark(text) 
        text = replaceMultiQuestionMark(text)
        text = replaceMultiStopMark(text)

        totalElongated += countElongated(text) 
        tokenized_tweet = tokenize(text)

        clean_tweets.append([tokenized_tweet, totalEmoticons, totalSlangs, 
                            totalSlangsFound, totalElongated, totalMultiExclamationMarks,
                            totalMultiQuestionMarks, totalMultiStopMarks, totalAllCaps])
    # form new dataframe
    df_clean_tweets = pd.DataFrame(clean_tweets,columns=['tokenized_tweet', 'totalEmoticons', 'totalSlangs', 
                            'totalSlangsFound', 'totalElongated', 'totalMultiExclamationMarks',
                            'totalMultiQuestionMarks', 'totalMultiStopMarks', 'totalAllCaps'])

    return df_clean_tweets

# def spellcheck(tweet):
#     tweet_spellchecked = []
#     print(len(tweet))
#     for word in tweet:
#         if len(word)>1:
#             word = spellCorrection(word) # Technique 12: correction of spelling errors
#         tweet_spellchecked.append(word)
#     return tweet_spellchecked

price_labels = pd.read_csv("../../Raw Data/Price/price_labels.csv")

for i in range(len(ticker_symbol)):
    df = pd.read_csv('../Raw Data/Tweets/'+ticker_symbol[i]+'_tweets.csv')
    print("Now cleaning:", ticker_symbol[i])

    print("Check pos tag...")
    if ticker_symbol[i] in ['JPM', "MMM", "KO", "JNJ", "PFE", "TRV", "V", "UNH"]:
        df_filter = df
    else:
        df_filter = spacy_pos(df, ticker_name[i])
        
    print("Group tweets by date...")
    df_filter = group_tweets_by_date(df, ticker_symbol[i], ticker_name[i])
    print("Number of records (weekdays):", df_filter.shape)
    
    print("Process raw tweets...")
    df_clean_tweets = clean_dirty_tweets(df_filter.text_removeCompany)
    
# #     spell_check_col = [spellcheck(tweet) for tweet in df_clean_tweets['tokenized_tweet']]
# #     print("spell check")
# #     df_clean_tweets['tokenized_tweet_spellcheck'] = spell_check_col
    
    # Join original df with df from tokenising + results
    df_tweets_final = pd.concat([df_filter, df_clean_tweets], axis = 1)
    
    ####################################################################
    ###  7. Join with prices, derive price features and target label ###
    ####################################################################
    price_labels_xticker = price_labels[price_labels['Ticker']==ticker_symbol[i]][['Date', "Adj Close"]]
    print("Number of business days:", price_labels_xticker.shape)
    price_labels_xticker.loc[:,'Date'] = pd.to_datetime(price_labels_xticker['Date']).dt.date
    price_labels_xticker.loc[:,'hist_returns'] = np.log10(price_labels_xticker['Adj Close']/price_labels_xticker['Adj Close'].shift())
    price_labels_xticker.loc[:,'returns5'] = np.log10(price_labels_xticker['Adj Close'].shift(-5)/price_labels_xticker['Adj Close'])
    price_labels_xticker.loc[:,'label5'] = np.where(price_labels_xticker['returns5']>=0,1,-1)
    
    joined_df = price_labels_xticker.join(df_tweets_final.set_index("stock_date"), on='Date', how='left')
    print("Longest NaN period:", joined_df.text.isnull().astype(int).groupby(joined_df.text.notnull().astype(int).cumsum()).sum().max())
#     joined_df = joined_df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    joined_df['Date'] = pd.to_datetime(joined_df['Date'])
    joined_df['Year'] = joined_df.Date.dt.year
    joined_df['Month'] = joined_df.Date.dt.month
    
    joined_df['vader_standardise'] = (joined_df['vader']-joined_df['vader'].expanding().mean())/joined_df['vader'].expanding().std()
    joined_df['vader3'] = joined_df['vader_standardise'].rolling(window=3, min_periods=2).sum()

    joined_df.to_pickle("../../Processed Data/Tweets/"+ticker_symbol[i]+"_df.pkl")
    
    
