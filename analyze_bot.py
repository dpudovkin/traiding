#!/usr/bin/env python

__author__ = "Danil Pudovkin @dapudovkin"

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

# Create parser for script
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--ticker", metavar="TICKER", type=str, help="Ticker to analyze")
parser.add_argument("-i", "--interval", metavar="INTERVAL", type=str, help="History interval")
parser.add_argument("-p", "--price-type", metavar="PRICE_TYPE", type=str, default="Adj Close",
    help="Price type to analyze: 'Open', 'High', 'Low', 'Close', 'Adj Close' (default).")
args = parser.parse_args()

ticker = args.ticker
interval = args.interval
price_type = args.price_type
end_date = datetime.now().date()
start_date = end_date - timedelta(days=729)

def get_ticker(ticker, start, end, interval):
    return yf.download(ticker, start, end, interval=interval)

# Get ticker quotes 
logger.info(f"Get data for {ticker}.")
td = pd.DataFrame(get_ticker(ticker, start_date, end_date, interval))

# mpf.plot(td)
# plt.show()

# Calculate MACD
logger.info(f"Build MACD indicator for {ticker}.")
td["EMA(26)"] = pd.DataFrame.ewm(td[price_type], span=26).mean()
td["EMA(12)"] = pd.DataFrame.ewm(td[price_type], span=12).mean()
td["MACD"] = td["EMA(12)"] - td["EMA(26)"]
td["SMA(9)"] = td["MACD"].rolling(9).mean()

# plt.plot(td.index, td[price_type])
# plt.plot(td.index, td["EMA(26)"], color="yellow")
# plt.plot(td.index, td["EMA(12)"], color="red")
# plt.show()

# plt.bar(td.index, td["MACD"])
# plt.plot(td.index, td["SMA(9)"], color="red")
# plt.show()

logger.info(f"Analyze {ticker} for deals.")
td["Transaction"] = [None for i in range(len(td.index))]

for i, time in enumerate(td.index):
    
    if (td["SMA(9)"][i] > td["MACD"][i] and 
            td["SMA(9)"][i-1] < td["MACD"][i] and 
            td["MACD"][i] > 0 and i != 0):
        td.at[time, "Transaction"] = "buy"
    elif (td["SMA(9)"][i] < td["MACD"][i] and 
            td["SMA(9)"][i-1] > td["MACD"][i] and 
            td["MACD"][i] < 0 and i != 0):
        td.at[time, "Transaction"] = "sell"
        
td = td.loc[td["Transaction"].isin(["buy", "sell"])]
td.to_excel("deals_report.xlsx")