# BT4222 Project - Text Analysis for Stock Returns Prediction

Increasingly, the use of textual data has become prevalent as they contain material information that is indicative of a company’s performance (Y Kim et al., 2014), and have potential to be valuable fundamental predictors on how stock prices will be affected. We hypothesise the effectiveness of using textual data as short term and long term drivers of stock returns to achieve superior returns from trading. Our project builds a universal model that predicts the daily returns of any Dow Jones Industrial Average (DJIA) company by combining the predictions of multiple textual data that use different returns horizons.

We use the time period of 1 Jan 2010 to 31 Dec 2017 as our training set for supervised learning, and 1 Jan 2018 to 31 Dec 2019 as our testing period. 5 of the 30 constituents of the Dow Jones Industrial Average (DJIA) - Caterpillar, Disney, Dow Inc, Travelers Companies and Walgreens Boots Alliance - are excluded due to data quality issues. We transform unstructured textual data into a structured, machine readable format, apply analytics to understand drivers behind stock returns, and make predictions based on these relationships using supervised machine learning. We use 5 different data sources:
1. __Annual reports (10K) and quarterly reports (10Q)__ capture longer term fundamental insights
2. __Event filings (8K)__ capture changing fundamentals in real time
3. __Financial news__ capture how news of fundamentals are interpreted and disseminated
4. __Tweets__ capture existing sentiments and public opinion
5. __Amazon reviews__ capture how well consumers react to company products which is indicative of the success of revenue streams. The Amazon product review is limited to Apple for our proof of concept.

## Setting Up
These instructions will get you a copy of the project up and running on your local machine. Please refer to <a href="https://github.com/jasmineseah-17/BT4222-Text-Analysis-For-Stock-Returns-Prediction/blob/master/Data%20Files%20Documentation.pdf" target="_blank">this documentation</a> for the description of each file.

### Download the Files
1. The zipped version of our data files are ~4 Gigabytes large and are not able to include it in this project. Hence, we provide a Google Drive link where you can download the Jupyter Notebooks as well as the data files in the correct format.
3. This repo contains the Jupyter Notebooks alone, you can use this google drive link `https://drive.google.com/open?id=1H14Us94oW12P2cFqY90FVIVeXw5jykr7` to obtain the full project with data and code, already formatted.

### Install the Required Packages, Lexicons, Model and Word Embeddings
1. Set up virtual environment by entering the following code in the terminal client:
```
conda create -n yourenvname python=3.6 anaconda
```
&ensp;&ensp;Then press `y` to proceeed. This will install the Python version, and all the associated anaconda packaged libraries at `path_to_your_anaconda_location/anaconda/envs/yourenvname`.
Activate the virtual environment as follows:
```
conda activate yourenvname
```
&ensp;&ensp;Then, use `requirements.txt` to install the correct versions of the required Python libraries to run the Python code we have written.
```
pip install -r requirements.txt
```
2. Download the following lexicons which we used to perform NLP. Simply run the code chunk below in your python notebook/ script.
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('vader_lexicon')
```
3. Download the following spacy model either on your python notebook/ script.
```
import spacy
!python -m spacy download en_core_web_sm
```
4. Finally, visit the following website: https://nlp.stanford.edu/projects/glove/ and download the file `glove.6B.zip` which contains the GloVe pre-trained word vectors used in our CNN Bi-directional LSTM model. Save this file in the following directory `Analytics/8K SEC/glove.6B`.

## Running the Codes
This is a 2-part project - (i) Obtain predictions from individual data sources for the 25 companies, (ii) Combine these predictions with technical and seasonal indicators for the final signal of the daily returns of any company.

### Individual Model for each Data Source
The 5 data sources (10K/Q, 8K, financial news, tweets, and Amazon reviews) individually predict the daily returns of the 25 DJIA companies. File naming conventions are standardised across all files. For more details, refer to <a href="https://github.com/jasmineseah-17/BT4222-Text-Analysis-For-Stock-Returns-Prediction/blob/master/Data%20Files%20Documentation.pdf" target="_blank">this documentation</a>. To build the model, we follow the steps in the Machine Learning Life Cycle, which include:
1. Data Extraction
2. Data Pre-processing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Building and Evaluation

### Meta-learner
We build 2 meta-learners - a company agnostic meta-learner (`final_metalearner.ipynb`), and a meta-learner specific to Apple Inc. (`aapl_metalearner.ipynb`). Model features include predictions from the data sources, technical indicators, seasonal indicators, and the sector the company belongs to. The company agnostic model aims to predict the direction of daily returns of any company in the DJIA index, thus does not include company-specific features. Predictions from Amazon reviews are only included for the Apple Inc. specific meta-learner.

For more details on the exact features used in the meta-learner, refer to `join_sources.ipynb` which concatenates all features to form the input for the meta-learners.

### Evaluate Meta-learner via Backtest
We use the following metrics to evaluate the meta-learners:

| __Data Science Metrics__ | __Business Metrics__  |
| :------- | :--- |
| Accuracy | Sharpe Ratio |
| F1-score | Annualised Returns |
| Precision | Annualised Volatility |
| Recall | Maximum Drawdown |

Our benchmark of a good model is one that beats the long-only investment returns. From the data science performance metrics, the long-only portfolio performs better. However, from a business perspective, both the company agnostic and AAPL specific meta-learners greatly outperform the long-only model. For the exact figures, refer to `Backtest.ipynb` and `Backtest-Apple.ipynb` respectively.

## Authors

* **Jasmine Seah** - [jasmineseah-17](https://github.com/jasmineseah-17)
* **Kaustubh Jagtap** - [ksjagtap](https://github.com/ksjagtap)
* **Nicklaus Ong Jing Xue** - [nicklausong](https://github.com/nicklausong)
* **Sung Zheng Jie** - [zhengjiesung](https://github.com/zhengjiesung)
* **Vienna Wong** - [viennawongjw](https://github.com/Viennawongjw)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
