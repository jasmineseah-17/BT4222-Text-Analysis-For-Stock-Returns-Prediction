# BT4222-NLP-Project

In our project, we test whether the use of Natural Language Processing is an effective proxy for both long- and short-term fundamental drivers that can potentially improve existing quantitative trading strategies in the DJIA space. We have excluded 5 of the 30 constituents of the Dow Jones Industrial Average (DJIA) - Caterpillar, Disney, Dow Inc, Travelers Companies and Walgreens Boots Alliance due to data quality issues. We transform unstructured textual data into a structured, machine readable format, apply analytics to understand drivers behind stock returns, and make predictions based on these relationships using machine learning. We used 5 different data sources - 
1. __Annual reports (10K) and quarterly reports (10Q)__ capture longer term fundamental insights
2. __Event filings (8K)__ capture changing fundamentals in real time
3. __Financial news__ capture how news of fundamentals are interpreted and disseminated
4. __Tweets__ capture existing sentiments and public opinion
5. __Amazon reviews__ capture how well consumers react to company products which is indicative of the success of revenue streams. The Amazon product review is limited to Apple for our proof of concept. 

By utilising sentiment analysis and NLP features from these textual data sources with other technical indicators, we aim to better predict the daily returns of companies. A decrease in sentiment obtained from the text data suggests a bleaker company outlook and potential, thus decreasing the stock price. Conversely, if sentiment is positive, the stock price is likely to increase.  We use the time period of 1 Jan 2010 to 31 Dec 2017 as our training set to optimise on, and 1 Jan 2018 to 31 Dec 2019 as our testing period. 

## Setting Up
These instructions will get you a copy of the project up and running on your local machine. Please refer to <a href="http://recordit.co/" target="_blank">this data_files_documentation.docx</a> for the description of each file.

### Download the Files
1. Create a folder to store all files. Ensure all code and data are stored in same folder.
2. The codes - Clone this repo to your local machine using `https://github.com/link`.
3. The data - Download everything in this <a href="http://recordit.co/" target="_blank">OneDrive folder</a>.

### Install the Required Packages and Lexicons
1. Use `requirement.txt` to install the correct versions of the required Python libraries to run the Python code we have written.
2. You will also need to download the following lexicons which we used to perform NLP. Simply run the code chunk below in your python notebook/ script.
``` 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## Running the Codes
This is a 2-part project - (i) Obtain predictions from individual data sources for the 25 companies, (ii) Combine these predictions with technical and seasonal indicators for the final signal of the daily returns of any company.

### Individual Model for each Data Source
The 5 data sources (10K/Q, 8K, financial news, tweets, and Amazon reviews) individually predict the daily returns of the 25 DJIA companies. File naming conventions are standardised across all files. For more details, refer to this documentation here. To build the model, we followed the steps in the Machine Learning Life Cycle, which include:
1. Data Extraction via `extraction.py`
2. Data Pre-processing via `cleaning.py`
3. Exploratory Data Analysis via `model_building.ipynb`
4. Feature Engineering via `model_building.ipynb`
5. Model Building and Evaluation via `model_building.ipynb`

### Metalearner
We built 2 metalearners - a company agnostic metalearner (`final_metalearner.ipynb`), and a metalearner specific to Apple Inc. (`aapl_metalearner.ipynb`). Model features include predictions from the data sources, technical indicators, seasonal indicators, and the sector the company belongs to. The company agnostic model aims to predict the direction of daily returns of any company in the DJIA index, thus does not include company-specific features. Predictions from Amazon reviews are only included for the Apple Inc. specific metalearner.

For more details on the exact features used in the metalearner, refer to `join_sources.ipynb` which concatenates all features to form the input for the metalearners.

## Authors

* **Jasmine Seah** - [jasmineseah-17](https://github.com/jasmineseah-17)
* **Kaustubh Jagtap** - [jasmineseah-17](https://github.com/nicklausong)
* **Nicklaus Ong Jing Xue** - [nicklausong](https://github.com/nicklausong)
* **Sung Zheng Jie** - [zhengjiesung](https://github.com/zhengjiesung)
* **Vienna Wong** - [Viennawongjw](https://github.com/Viennawongjw)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
