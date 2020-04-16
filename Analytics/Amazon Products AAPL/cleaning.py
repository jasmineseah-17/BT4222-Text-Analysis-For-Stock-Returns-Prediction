import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
import datetime
from datetime import datetime, timedelta


# # Get Relevant Reviews from both aapl_public.tsv and aapl_amazon_csv
# 1. aapl_public.txv is aws public dataset. Wireless products dataset was picked after reviewing through other relevant datasets, only wireless products dataset contains Apple Products
# 2. aapl_amazon_csv is amazon reviews that was scraped using scrapy by going Amazon Apple products
# 3. Main part here is to get reviews relevant to Apple Company and from 2010-2019
# 4. Apple Products Dates is a dataset of Apple Products and its Release Dates that is found online (https://en.wikipedia.org/wiki/Timeline_of_Apple_Inc._products) & (https://941am.com/) as reference
# 5. This dataset will help to filter and retrieve relevant products on both aapl_public and aapl_amazon datasets


df_wireless= pd.read_csv('../../Raw Data/Amazon Product AAPL/aapl_public.tsv', sep='\t',
                                usecols=['marketplace', 'customer_id', 'review_id',
                                        'product_id', 'product_parent', 'product_title',
                                        'product_category', 'star_rating', 'helpful_votes',
                                        'total_votes', 'vine', 'verified_purchase', 'review_headline',
                                        'review_body', 'review_date'])



#get only apple related products by looking at the product title
#get products where the review dates are 2010 and above

df_wireless['product_title'] = df_wireless['product_title'].str.lower()
df_wireless= df_wireless[df_wireless['product_title'].notna()]
df_wireless_apple = df_wireless[df_wireless['product_title'].str.contains("apple")]
df = df_wireless_apple[df_wireless_apple['review_date']> '2010-01-01']


df = pd.read_excel('aapl_public_filtered.xlsx')



#get relevant variables
df = df[['product_title','review_id', 'product_id','product_category', 'star_rating', 'review_headline', 'review_body', 'review_date']]


#remove numbers and special characters
df["review_body"] = df['review_body'].replace("[^a-zA-Z]+", " ", regex = True)
df["review_headline"] = df['review_headline'].replace("[^a-zA-Z]+", " ", regex = True)

#remove white spaces
df["review_body"] = df['review_body'].str.strip()
df["review_headline"] = df['review_headline'].str.strip()



#fill review body na with empty string
df['review_body'].fillna('', inplace= True)
df = df[df['review_body'].notnull()]


#APPLE PRODUCTS AND ITS RELEASE DATES. Taken from https://941am.com/ 
#https://en.wikipedia.org/wiki/Timeline_of_Apple_Inc._products

products_dates = pd.read_excel('../../Raw Data/Amazon Product AAPL/aapl_products_dates.xlsx')



#Clean the product names and products where the release date is 2015 and below 

products_dates["new"] = products_dates["Product Name"].replace("[^a-zA-Z0-9]+", " ", regex = True)
products_dates['new'] = products_dates['new'].str.lower()
products_dates['new'] = products_dates['new'].str.replace('gen', 'generation')
products_2015 = products_dates[products_dates['Date'] < " 2016-01-01"]



#create product dates dictionary
pd_dates = pd.Series(products_2015['Date'].values,index=products_2015['new']).to_dict()
pd_dates = OrderedDict(sorted(pd_dates.items(), key=lambda x:x[1], reverse=True))



#creates a list of product names
p_list =list(products_dates['new'].str.strip())
pattern = '|'.join(p_list)


#take products only if product title contains the product names
df_new = df[df.product_title.str.contains(pattern)]

#a list of words that should not be included in the product titles
#remove those rows where product titles contains np_list

np_list = ['earphones','earphone', 'earbuds','bluetooth', 'case', 'cable','speaker', 'portable', 'headphones', 'headset',
          'bluetooth', 'protector', 'samsung', 'android', 'adapter', 'usb', 'charger', 'earbud', 'cover', 'hdmi',
          'stand','leather','replacement', 'mount', 'holder', 'battery', 'mounting', 'sticker', 'replaceable',
          'bumper','len', 'packaging', 'package', '/', 'armband', 'frame', 'stylus', 'band', 'digitizer', 'charging','cleaner',
          'display', 'skin','kit','handset', 'set', 'strap','headphone', 'accessory', 'decal', 'wallet', 'bag', 'pouch',
          'mp3', 'mp3s', 'adaptor','plug', 'shell', 'cellet','cloths', 'cloth', '3102mss','selfiepod','tool', 'shield',
          'shock', 'armor', 'film', 'protection', 'sim', 'plastic','tripod', 'car','cradle','tempered','design','invisibleshield',
          ]
remove = '|'.join(np_list)
df_new_wl = df_new[~df_new.product_title.str.contains(remove)]


df_new_wl['product_title'] = df_new_wl['product_title'].replace("[^a-zA-Z0-9]+", " ", regex = True)


def like_function(x):
    ''' 
    return the release date of the product if the product_title 
    of the dataset contains the product in pd_dates dict
    
    '''
    date = ""
    for key in pd_dates: #key = product_title
        if key in x:
            date = pd_dates[key]
            break
    return date

def u_like_function(x):
    ''' 
    return the product name of the product if the product_title 
    of the dataset contains the product in pd_dates dict
    
    '''    
    product = ""
    for key in pd_dates:
        if key in x:
            product = key
            break
    return product

# Create new variables release date (of product) and unique products (that comes from Apple Product Dates)
df_new_wl['release_date'] = df_new_wl.product_title.apply(like_function)
df_new_wl['unique_products'] = df_new_wl.product_title.apply(u_like_function)


#SCRAPED AMAZON REVIEWS
amazon_scrape = pd.read_csv("../../Raw Data/Amazon Product AAPL/aapl_amazon.csv",encoding = "ISO-8859-1")
amazon_scrape['product name'] = amazon_scrape['product name'].str.lower()

#Keep products where product names only contain the product names in the Apple Products Dates
df_am_scr = amazon_scrape[amazon_scrape['product name'].str.contains(pattern)]

#a list of words that should not be included in the product titles
#remove those rows where product titles contains anp_list

anp_list = ['/', 'mount', 'cider', 'case', 'onion', 'delivery','cable','water', 'adapter','smart', 'earphones', 'guide',
           'remote', 'protector']
remove = '|'.join(anp_list)
df_am_scr = df_am_scr[~df_am_scr['product name'].str.contains(remove)]

#clean the product name
df_am_scr['product name'] = df_am_scr['product name'].replace("[^a-zA-Z0-9]+", " ", regex = True)
df_am_scr.reset_index(inplace = True, drop=True)


#Get products and dates as dictionary

pd_dates_2 = pd.Series(products_dates['Date'].values,index=products_dates['new']).to_dict()
pd_dates_2 = OrderedDict(sorted(pd_dates_2.items(), key=lambda x:x[1], reverse=False))



def like_function(x):
    ''' 
    return the release date of the product if the product_title 
    of the dataset contains the product in pd_dates_2 dict
    
    '''
    date = ""
    for key in pd_dates_2:
        if key in x:
            date = pd_dates_2[key]
            break
    return date

def u_like_function(x):
    ''' 
    return the product name of the product if the product_title 
    of the dataset contains the product in pd_dates_2 dict
    
    '''    
    product = ""
    for key in pd_dates_2:
        if key in x:
            product = key
            break
    return product

df_am_scr['release_date'] = df_am_scr['product name'].apply(like_function)
df_am_scr['unique_products'] = df_am_scr['product name'].apply(u_like_function)

#remove numbers and special characters
df_am_scr['comment'] = df_am_scr['comment'].replace("[^a-zA-Z]+", " ", regex = True)
#df["review_headline"] = df['review_headline'].replace("[^a-zA-Z]+", " ", regex = True)

#remove white spaces
df_am_scr['comment'] = df_am_scr['comment'].str.strip()
#df["review_headline"] = df['review_headline'].str.strip()
df_am_scr['comment'].fillna('', inplace= True)

df_new_wl_2 = df_new_wl[['product_title', 
       'star_rating', 'review_body', 'review_date',
       'release_date', 'unique_products']]


df_apple = pd.concat([df_new_wl_2,df_am_scr.rename(columns={'product name':'product_title', 
                                                           'comment': 'review_body',
                                              'stars' : 'star_rating',
                                                           'date': 'review_date'})], ignore_index=True)



#Clean datetime format
#change all datetime to the same string format
for index, row in df_apple.iterrows():
    date = row['review_date']
    
    print(date)
    if len(date) > 10:
        new = datetime.strptime(date, '%B %d, %Y')
        df_apple.at[index,'review_date'] = str(new.year)+ '-'+ str(new.month)+ '-' + str(new.day)
        print(row['review_date'])



df_apple.sort_values('review_date')

# index 4509 date is wrong, drop that row
df_apple.loc[4509]
df_apple.drop([4509], inplace = True)


#string date to datetime object
df_apple['review_date'] = df_apple['review_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_apple


# ## Map Products Release Dates and take Top 5 latest products based on review dates
# 1. Make sure that the products mapped from the Apple Products Dates and Release dates is up to date with the products in the Amazon Reviews Dataset
# 2. Keep the reviews only if the product is in the top 5 latest products based on review dates and release dates. This is to ensure that the review is not too outdated. 
# 
# For example, if the product review is iPhone 4 which is release in 2010, and the review date is 2015, it will not be accurate in telling us Apple's Sales/Performance in 2015 as the phone was released 5 years ago. So to account for that, we are only going to take the top 5 latest products based on their release dates and map to the review dates and drop other outdated reviews 


products_dates["new"] = products_dates["Product Name"].replace("[^a-zA-Z0-9]+", " ", regex = True)
products_dates['new'] = products_dates['new'].str.lower()
products_dates['new'] = products_dates['new'].str.replace('gen', 'generation')



##Get products and dates as dictionary
pd_dates = pd.Series(products_dates['Date'].values,index=products_dates['new']).to_dict()
pd_dates = OrderedDict(sorted(pd_dates.items(), key=lambda x:x[1], reverse=False))
pd_dates



#To ensure that product mapped from the Apple Product Dates and the Amazon reviews dataset is up to date

def map_product_function(row):
    '''
    This function is to ensure that the unique product name that comes from the Apple Product Dates is mapped 
    to the most relevant product based on the review date and the release date of the product
    
    Takes in each product of the Amazon Review dataset
    Create a new_dict with all the products in Apple Product Dates that contains the product title of
    the Amazon Reviews dataset. 
    e.g. product title = apple iphone 4s, review date = 2013-04-17 00:00:00 (from Amazon Review Dataset)
    new_dict = [['iphone 4', Timestamp('2010-06-24 00:00:00')], ['iphone 4s', Timestamp('2011-10-14 00:00:00')]] (Apple Products Dates)
    
    Take the product with the least difference between the review date and the product release date
    unique product, release date = 'iphone 4s', Timestamp('2011-10-14 00:00:00')
    
    
    '''
    date = row['review_date']
    product_title = row['product_title']
    unique_pro = ''
    release_date = ''
    new_dict = []
    min_time = 1000000000000
    for key in pd_dates:
        if key in product_title and pd_dates[key] < date:
            key_list = []
            key_list.append(key)
            key_list.append(pd_dates[key])
            new_dict.append(key_list)
    print('products in new_dict: ' , new_dict)
    if not new_dict:
        return unique_pro, release_date
    else:
        unique_pro = new_dict[0][0]
        release_date = new_dict[0][1]
        for i in range (0, len(new_dict)):
            diff = (date - new_dict[i][1]).days
            if min_time > diff:
                min_time = diff
                unique_pro = new_dict[i][0]
                release_date = new_dict[i][1]
        print('product_title, unique_pro, release_date, date') 
        print(product_title, unique_pro, release_date, date)    
        return unique_pro, release_date

df_apple[['test_pro', 'test_date']] = df_apple.apply(lambda row: pd.Series(map_product_function(row)), axis=1)
df_apple['diff'] = df_apple.apply(lambda row: 1 if row['unique_products'] == row['test_pro'] else 0, axis = 1)

new_df = df_apple.copy()

#Replace the orig unique products and the release dates to the updated one
new_df['release_date'] = new_df['test_date']
new_df['unique_products'] = new_df['test_pro']
del new_df['test_pro']
del new_df['test_date']
del new_df['release_date']
del new_df['diff']


#since not all of the products in the Amazon reviews are in the product dates dataset
#keep the Apple Products Dates with products that are in Amazon Reviews Dataset

pro = list(new_df.unique_products.unique())
products_dates =products_dates[products_dates['new'].isin(pro)]
products_dates['product_tags'] = ''
products_dates =products_dates.reset_index()

# top 5 latest products and tag it with each product

for i in range(4, len(products_dates)):
    p1 = products_dates['new'][i-4]
    p2 = products_dates['new'][i-3]
    p3 = products_dates['new'][i-2]
    p4 = products_dates['new'][i-1]
    p5 = products_dates['new'][i]
    new_list = [p1,p2,p3,p4,p5]
    products_dates.at[i, 'product_tags'] = new_list
    

products_dates.at[0,'product_tags'] = [products_dates['new'][0]]
products_dates.at[1, 'product_tags'] = [products_dates['new'][0], products_dates['new'][1]]
products_dates.at[2, 'product_tags'] = [products_dates['new'][0], products_dates['new'][1], products_dates['new'][2]]
products_dates.at[3, 'product_tags'] = [products_dates['new'][0], products_dates['new'][1], products_dates['new'][2], products_dates['new'][3]]

new_df = new_df.reset_index()

#for each review dates find the top 5 latest products and tag with it
#check if review date falls in between which two products release date in Apple Products Dates
# take the top 5 latest products from p1 where p1 < review_date < p2
new_df['product_tags'] = ''
for j in range(0, len(new_df)):
    review_date = new_df['review_date'][j]
    product_list = []
    for i in range(0, len(products_dates)):
        if i == len(products_dates)-1:
            continue
        else:
            p1 = products_dates['Date'][i]
            p2 = products_dates['Date'][i+1]
            if p1 < review_date < p2:
                new_list = products_dates['product_tags'][i]
                new_df.at[j, 'product_tags'] = new_list
                continue

#make sure review date is before 2020 (our problem statement 2010-2019)
new_df = new_df[new_df.review_date < '2020-01-01']

#make sure that the product is in the product_tags -> fulfil the criteria of the product being the latest top 5
new_df['filter'] = new_df.apply(lambda row: 1 if row['unique_products'] in row['product_tags'] else 0, axis=1)
df_filtered = new_df[new_df['filter'] == 1]
df_filtered

del df_filtered['index']
del df_filtered['filter']

#clean the stars!
df_filtered['stars'] = df_filtered['star_rating'].astype(str).str[0]
df_filtered['stars'] = df_filtered['stars'].apply(lambda x: int(x))
df_filtered
del df_filtered['star_rating']

df_filtered.to_pickle("aapl_df.pkl")



# %%
