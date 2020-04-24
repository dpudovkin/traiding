#!/usr/bin/env python3

__author__ = "Danil Pudovkin @dapudovkin"

import csv
import json
import logging
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf

from datetime import datetime as dt, timedelta
from functools import reduce
from sys import argv, exit
from glob import glob

# Get input params
with open("./params.json", "r") as file:
    params = json.load(file)


origin_balance = params["balance"]
broker_fee = params["broker_fee"]
period = params["period"]
tf_entry = params["tf_entry"]
tf_exit = params["tf_exit"]
short_ema = params["short_ema"]
long_ema = params["long_ema"]
signal_sma = params["signal_sma"]
k_open = params["k_open"]
k_close = params["k_close"]

balance = origin_balance

btcusd_30m_path = "./BTCUSDT/"
ticker = "BTCUSD=X"
price_type = "Close"
end_date = dt.now()
start_date = end_date - timedelta(days=period)

orders = {
    "order_type": [],
    "open_time": [],
    "open_price": [],
    "close_time": [],
    "close_price": [],
    "net_income": []
}

equity = {
    "balance": [],
    "datetime": []
}


# Set logging parameters
log_format = '%(asctime)s %(name)s: %(levelname)s: %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
script_name = argv[0].split('/')[-1]
logging.basicConfig(
    format=log_format, level=logging.INFO, datefmt=date_format)

# Create logging instance
logger = logging.getLogger(script_name)


def str_to_datetime(x):
    return dt.strptime(x, "%Y-%m-%dT%H:%M:%S")

def to_datetime(x):
    return dt.fromtimestamp(x.timestamp())

def update_equity(balance, datetime):
    equity["balance"].append(balance)
    equity["datetime"].append(datetime)

def get_fee(delta):
    return abs(delta) * broker_fee

def update_balance(delta):
    global balance
    balance += delta

def get_ticker(ticker, start, end, interval, source="url", path=None):
    """
    Params:
        source: get ticer data from the specified source
        path: path with csv files with the ticker data
    """

    logger.info(f"Get data for {ticker} with {interval} interval")
    if source == "url":
        df = yf.download(ticker, start, end, interval=interval).tz_localize(None)
        df["Open"] = list(map(lambda x: np.round(x, 2), df["Open"]))
        df["High"] = list(map(lambda x: np.round(x, 2), df["High"]))
        df["Low"] = list(map(lambda x: np.round(x, 2), df["Low"]))
        df["Close"] = list(map(lambda x: np.round(x, 2), df["Close"]))
        return df
    elif source == "files" and isinstance(path, str):
        data = []
        time = []
        for file in sorted(glob(path + "*.csv")):
            with open(file, "r") as f:
                for row in csv.reader(f):
                    row = row[0].split(";")
                    if start < str_to_datetime(row[0]) < end:
                        data.append(list(map(lambda x: float(x), row[1:])))
                        time.append(str_to_datetime(row[0]))
        df =  pd.DataFrame(data, index=time,
            columns=["Open", "High", "Low", "Close", "Volume"]) 
        df.index.name = "Datetime"
        return df

def build_macd(df, tf):
    df["EMA(26)"] = pd.DataFrame.ewm(df[price_type], span=long_ema).mean()
    df["EMA(12)"] = pd.DataFrame.ewm(df[price_type], span=short_ema).mean()
    df["MACD"] = df["EMA(12)"] - df["EMA(26)"]
    df["SMA(9)"] = df["MACD"].rolling(signal_sma).mean()
    logger.info(f"Build MACD indicator for {ticker} with {tf} interval.")
    return df

def open_order(order_type, open_time, open_price):
    orders["order_type"].append(order_type)
    orders["open_time"].append(open_time)
    orders["open_price"].append(open_price)
    # Return order ID (index)
    logger.info(f"Open order with price {open_price} at {open_time}.")
    return len(orders["order_type"]) - 1

def close_order(n, close_time, close_price):
    orders["close_time"].append(close_time)
    orders["close_price"].append(close_price)

    if orders["order_type"][n] == "buy":
        delta = orders["close_price"][n] - orders["open_price"][n]
    elif orders["order_type"][n] == "sell":
        delta = orders["open_price"][n] - orders["close_price"][n]

    net_income = delta - get_fee(delta)
    orders["net_income"].append(net_income)
    update_balance(net_income)
    update_equity(balance, close_time)
    logger.info(f"Close order with price {close_price} at {close_time}. "
        f"Profit/Loss: {net_income} Balance: {balance}")

def open_cond1(df, i):
    if df["MACD"][i] < 0:
        return True
    return False

def open_cond2(df, i):
    if (df["MACD"][i-1] < df["MACD"][i] > df["MACD"][i+1] and
            df["MACD"][i] > 0):
        return True
    return False

def open_cond3(t1, t2):
    if (t1["MACD"] >= t2["MACD"] and
            t1["MACD"] > 0 and t2["MACD"] > 0):
        return True
    return False

def open_cond4(t1, t2):
    if (t1["Close"] > t1["Open"] and 
            t1["Close"]*(1-k_open) <= t2["High"]):
        return True
    elif (t1["Close"] <= t1["Open"] and 
            t1["Open"]*(1-k_open) <= t2["High"]): 
        return True
    return False

def close_cond1(df, i):
    if (df["MACD"][i-1] > df["MACD"][i] < df["MACD"][i+1] and
            df["MACD"][i-1] < 0 and df["MACD"][i] < 0):
        return True
    return False

def close_cond2(t1, t2):
    if (t1["MACD"] <= t2["MACD"] and
            t1["MACD"] < 0 and t2["MACD"] < 0):
        return True
    return False

def close_cond3(t1, t2):
    if (t1["Close"] > t1["Open"] and
            t1["Open"]*(1+k_close) >= t2["Low"]):
        return True
    elif (t1["Close"] <= t1["Open"] and
            t1["Close"]*(1+k_close) >= t2["Low"]): 
        return True
    return False


update_equity(balance, start_date)
# Get tickers quotes with TF tf_entry
df_entry = get_ticker(ticker, start_date, end_date, tf_entry)
# Get ticker quotes with TF tf_exit
df_exit = get_ticker(
    ticker, start_date, end_date,
    tf_exit, source="files", path=btcusd_30m_path)

# Build MACD with TF tf_entry
df_entry = build_macd(df_entry, tf_entry)
# Build MACD with TF tf_exit
df_exit = build_macd(df_exit, tf_exit)

for i, _ in enumerate(df_entry.index):
    # Check open condition #1
    if open_cond1(df_entry, i):
        # Move back from MACD < 0 point to find t1, t2
        points = []
        for k in range(i-1, -1, -1):
            if k == i-1:
                continue
            if open_cond2(df_entry, k):
                points.append(df_entry.index[k])
                if len(points) == 2:
                    points = list(map(lambda x: df_entry.loc[x], points))
                    break
        if len(points) < 2:
            continue
        t1, t2 = points
        if open_cond3(t1, t2):
            if open_cond4(t1, t2):
                open_time = df_entry.index[i+1]
                open_price = df_entry["Open"][i+1]
                n = open_order("sell", str(to_datetime(open_time)), open_price)

                for j, _ in enumerate(df_exit.index):
                    if df_exit.index[j] > open_time:
                        # Here we move back from the point with the time
                        # that greater than open time
                        points = []
                        for k in range(j, i+1, -1):
                            if k == j:
                                continue
                            if close_cond1(df_exit, k):
                                points.append(df_exit.index[k])
                                if len(points) == 2:
                                   points = list(map(lambda x: df_exit.loc[x], points)) 
                                   break
                        if len(points) < 2:
                            continue
                        t1, t2 = points
                        if close_cond2(t1, t2):
                            if close_cond3(t1, t2):
                                close_time = df_exit.index[j+1]
                                close_price = df_exit["Open"][j+1]
                                close_order(n, str(to_datetime(close_time)), close_price)
                                break

orders = pd.DataFrame(orders)
pos_orders = orders[orders["net_income"] > 0]
neg_orders = orders[orders["net_income"] < 0]
# Average profit per positive order
avg_pppo = reduce(lambda x,y: x+y, pos_orders["net_income"])/len(pos_orders.index)
# Average loss per negative order
avg_lpno = reduce(lambda x,y: x+y, neg_orders["net_income"])/len(neg_orders.index)
# % of positive orders
rel_po = len(pos_orders.index)/len(orders.index)*100
# % of negative orders
rel_no = len(neg_orders.index)/len(orders.index)*100
annual_return = (balance - origin_balance)/(origin_balance * period) * 100 * 365
print("Count trades: ", len(orders.index))
print("Count pos trades: ", len(pos_orders.index))
print("Count neg trades: ", len(neg_orders.index))
print("Relative pos trades (%): ", rel_po)
print("Relative neg trades (%): ", rel_no)
print("Avg profit per pos trade:", avg_pppo)
print("Avg loss per neg trade:", avg_lpno)
print("Annual return (%): ", annual_return)
writer = pd.ExcelWriter("orders2018_2020.xlsx", engine="xlsxwriter")
orders.to_excel(writer, sheet_name="Orders 2018-2020")
writer.save()
equity = pd.DataFrame(equity["balance"], index=equity["datetime"], columns=["Equity"])
equity.plot()
plt.show()
