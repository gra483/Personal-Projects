import tweepy as tw
import pandas as pd
import torch
import json
import os
import requests
from lxml import html
from Twitter.Stocks.classifier_model import predict_sentiment
from Twitter.Stocks.classifier_model import BERTGRUSentiment
from collections import OrderedDict
from datetime import date
from transformers import BertTokenizer, BertModel


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


def query_twitter_api(json_file_name):
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Define the search term and the date_since date as variables
    with open(json_file_name, 'r') as json_file:
        data = json_file.read()
    raw_json = json.loads(data)["stocks"]
    curr_date = date.today().strftime("%Y/%m/%d")
    todays_data = {}

    for name, symbol in raw_json.items():
        #collect 200 tweets for every stock in the json file
        tweets_name = tw.Cursor(api.search, q=name, lang="en", since=curr_date).items(100)
        tweets_symbol = tw.Cursor(api.search, q=symbol, lang="en", since=curr_date).items(100)
        stock_text_data = [tweet.text for tweet in tweets_name]
        for tweet in tweets_symbol:
            if(tweet not in stock_text_data):
                stock_text_data.append(tweet.text)
        todays_data[name] = stock_text_data
    return todays_data


def classify_data(todays_data, model):
    #use pre-trained model
    sentiments = {}
    for stock, tweets in todays_data.items():
        sentiments[stock] = {"neg": 0, "pos": 0}
        for tweet in tweets:
            proba = predict_sentiment(model, tweet)
            if(proba <= 0.5): sentiments[stock]["neg"] = sentiments[stock]["neg"]+1
            else: sentiments[stock]["pos"] = sentiments[stock]["pos"]+1
        sentiments[stock]["pos_rate"] = sentiments[stock]["pos"]/(sentiments[stock]["pos"]+sentiments[stock]["neg"])
    return sentiments


def get_headers():
    return {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-GB,en;q=0.9,en-US;q=0.8,ml;q=0.7",
            "cache-control": "max-age=0",
            "dnt": "1",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}


def parse(ticker):
    url = "http://finance.yahoo.com/quote/%s?p=%s" % (ticker, ticker)
    response = requests.get(url, verify=False, headers=get_headers(), timeout=30)
    parser = html.fromstring(response.text)
    summary_table = parser.xpath(
        '//div[contains(@data-test,"summary-table")]//tr')
    summary_data = OrderedDict()
    other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(
        ticker)
    summary_json_response = requests.get(other_details_json_link)
    try:
        json_loaded_summary = json.loads(summary_json_response.text)
        summary = json_loaded_summary["quoteSummary"]["result"][0]
        y_Target_Est = summary["financialData"]["targetMeanPrice"]['raw']
        earnings_list = summary["calendarEvents"]['earnings']
        eps = summary["defaultKeyStatistics"]["trailingEps"]['raw']
        datelist = []

        for i in earnings_list['earningsDate']:
            datelist.append(i['fmt'])
        earnings_date = ' to '.join(datelist)

        for table_data in summary_table:
            raw_table_key = table_data.xpath(
                './/td[1]//text()')
            raw_table_value = table_data.xpath(
                './/td[2]//text()')
            table_key = ''.join(raw_table_key).strip()
            table_value = ''.join(raw_table_value).strip()
            summary_data.update({table_key: table_value})
        summary_data.update({'1y Target Est': y_Target_Est, 'EPS (TTM)': eps,
                             'Earnings Date': earnings_date, 'ticker': ticker,
                             'url': url})
        return summary_data
    except ValueError:
        print("Failed to parse json response")
        return {"error": "Failed to parse json response"}
    except:
        return {"error": "Unhandled Error"}


def get_price_data(json_file_name):
    #scrape price data for listed stocks
    with open(json_file_name, 'r') as json_file:
        data = json_file.read()
    raw_json = json.loads(data)['stocks']
    price_data = {}
    for name, symbol in raw_json.items():
        scraped_data = parse(symbol)

        price_data[name] = scraped_data["Previous Close"]
    return price_data


def dump_data(sentiments, prices):
    if(os.path.exists("stock_sentiments.csv")):
        csv_df = pd.read_csv("stock_sentiments.csv")
        day = csv_df["day"][len(csv_df)-1]+1
    else:
        csv_df = pd.DataFrame(columns=["pos_rate", "close", "name", "day"])
        day=0

    for name in prices.keys():
        # add row entry for each stock
        # COLS: Pos_rate, Close, Name, Day?
        d = {"pos_rate": [sentiments[name]["pos_rate"]], "close": [float(prices[name].replace(',', ''))],
             "name": [name], "day": [day]}
        df = pd.DataFrame(d)
        csv_df = csv_df.append(df, ignore_index=True)
    csv_df.to_csv("stock_sentiments.csv", index=False)


def main():
    #define and load model
    bert = BertModel.from_pretrained('bert-base-uncased')
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    model = BERTGRUSentiment(bert,
                             HIDDEN_DIM,
                             OUTPUT_DIM,
                             N_LAYERS,
                             BIDIRECTIONAL,
                             DROPOUT)
    model.load_state_dict(torch.load('tut6-model.pt'))
    #throw this part below on a daily clock on a server
    tweets = query_twitter_api("stocks.json")
    sentiments = classify_data(tweets, model)
    prices = get_price_data("stocks.json")
    dump_data(sentiments, prices)


if __name__ == '__main__':
    main()