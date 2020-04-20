#!/usr/bin/env python3

__author__ = "Danil Pudovkin @dapudovkin"

import json
import logging
import argparse
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf

from datetime import datetime, timedelta
from sys import argv, exit

# Set logging parameters
log_format = '%(asctime)s %(name)s: %(levelname)s: %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
script_name = argv[0].split('/')[-1]
logging.basicConfig(
    format=log_format, level=logging.INFO, datefmt=date_format)

# Create logging instance
logger = logging.getLogger(script_name)

# Get input params
with open("./params.json", "r") as file:
    params = json.load(file)

balance = params["balance"]
broker_fee = params["broker_fee"]
tf_entry = params["tf_entry"]
tf_exit = params["tf_exit"]
short_ema = params["short_ema"]
long_ema = params["long_ema"]
signal_sma = params["signal_sma"]

ticker = "BTCUSD=X"
price_type = "Close"
end_date = datetime.now().date()
start_date = end_date - timedelta(days=5)

deals = {
    "order_type": [],
    "open_time": [],
    "open_price": [],
    "close_time": [],
    "close_price": [] 
}

equity = []

def update_equity(balance):
    equity.append(balance)

def get_equity():
    print(equity)

def update_balance(delta):
    global balance
    if delta < 0:
        fee = abs(delta) * broker_fee
    else:
        fee = delta * broker_fee

    balance += delta - fee

def get_balance():
    print(balance)

def get_ticker(ticker, start, end, interval):
    return yf.download(ticker, start, end, interval=interval)

def build_macd(td, tf):
    logger.info(f"Build MACD indicator for {ticker} with {tf} interval.")
    td["EMA(26)"] = pd.DataFrame.ewm(td[price_type], span=long_ema).mean()
    td["EMA(12)"] = pd.DataFrame.ewm(td[price_type], span=short_ema).mean()
    td["MACD"] = td["EMA(12)"] - td["EMA(26)"]
    td["SMA(9)"] = td["MACD"].rolling(signal_sma).mean()

    return td

def open_order(order_type, open_time, open_price):
    logger.info(f"Open order with price {open_price} at {open_time}.")
    deals["order_type"].append(order_type)
    deals["open_time"].append(open_time)
    deals["open_price"].append(open_price)
    # Return order ID (index)
    return len(deals["order_type"]) - 1

def close_order(n, close_time, close_price):
    logger.info(f"Close order with price {close_price} at {close_time}.")
    deals["close_time"].append(close_time)
    deals["close_price"].append(close_price)

    if deals["order_type"][n] == "buy":
        delta = deals["close_price"][n] - deals["open_price"][n]
    elif deals["order_type"][n] == "sell":
        delta = deals["open_price"][n] - deals["close_price"][n]

    update_balance(delta)
    update_equity(balance)

update_equity(balance)
    
# Get ticker quotes 
logger.info(f"Get data for {ticker} with {tf_entry} interval")
td_entry = pd.DataFrame(get_ticker(ticker, start_date, end_date, tf_entry))
logger.info(f"Get data for {ticker} with {tf_exit} interval")
td_exit = pd.DataFrame(get_ticker(ticker, start_date, end_date, tf_exit))

# Build MACD
td_entry = build_macd(td_entry, tf_entry)
td_exit = build_macd(td_exit, tf_exit)

order_id = open_order("sell", "2020-04-20 11:00", 9000)
close_order(order_id, "2020-04-20 12:00", 8000)
get_balance()
get_equity()

# mpf.plot(td)
# plt.show()

# plt.plot(td.index, td[price_type])
# plt.plot(td.index, td["EMA(26)"], color="yellow")
# plt.plot(td.index, td["EMA(12)"], color="red")
# plt.show()

# plt.bar(td.index, td["MACD"])
# plt.plot(td.index, td["SMA(9)"], color="red")
# plt.show()

# logger.info(f"Analyze {ticker} for deals.")
# td["Transaction"] = [None for i in range(len(td.index))]

# for i, time in enumerate(td.index):
    
#     if (td["SMA(9)"][i] > td["MACD"][i] and 
#             td["SMA(9)"][i-1] < td["MACD"][i] and 
#             td["MACD"][i] > 0 and i != 0):
#         td.at[time, "Transaction"] = "buy"
#     elif (td["SMA(9)"][i] < td["MACD"][i] and 
#             td["SMA(9)"][i-1] > td["MACD"][i] and 
#             td["MACD"][i] < 0 and i != 0):
#         td.at[time, "Transaction"] = "sell"
        
# td = td.loc[td["Transaction"].isin(["buy", "sell"])]
# td.to_excel("deals_report.xlsx")