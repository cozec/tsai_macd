"""
https://github.com/matplotlib/mplfinance/blob/master/examples/panels.ipynb
macd_nd or zero lag macd is non-causal, giving too good results
macd causal is the only one we can get at current date
"""
import yfinance as yf
from yahoo_fin import stock_info as si
import mplfinance as mpf
import talib as ta
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2f}%'.format
import math
import numpy as np
from scipy.signal import lfilter, filtfilt
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, date, timedelta

import time
import pickle
import csv
from sklearn.preprocessing import MinMaxScaler

from parameters import *

import sys
sys.path.insert(0, "../backtesting_zipline")
from tutorial_trade_macd import detect_macd_signals, print_performance_summary
from zl_macd_basic_list import save_sp500_tickers

def get_MACD(df):
    """
    get MACD by pandas functions such as pandas.DataFrame.ewm
    get MACD no delay by filtfilt --> non-causal, need re-calculate for each past date

    data["macd"], data["macd_signal"], data["macd_hist"] = ta.MACD(data['Close'])
    """

    #First use Pandas to calculate the 12 period and 26 period exponential moving averages:
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()

    #The MACD Line is defined as the difference between these two moving averages:
    macd = exp12 - exp26
    macd_percentage = (exp12 - exp26) / exp26

    """
    The MACD Signal is defined as the 9 period exponential moving average of the MACD Line:
    We also calculate the difference between the MACD Line and the MACD Signal which we will plot as a histogram:
    """
    signal    = macd.ewm(span=9, adjust=False).mean()
    signal_percentage    = macd_percentage.ewm(span=9, adjust=False).mean()

    histogram = macd - signal
    histogram_percentage = macd_percentage - signal_percentage

    df['macd_hist'], df['macd'], df['macd_signal'] = histogram, macd, signal
    df['macd_hist_percentage'], df['macd_percentage'], df['macd_signal_percentage'] = \
        histogram_percentage, macd_percentage, signal_percentage

    return df


# cannot handle constant input
def simple_minmax_scaler(X):
    mn, mx = X.min(), X.max()
    X_scaled = (X - mn) / (mx - mn)
    return X_scaled

def array_MinMaxScaler(data): # wrong for array input!!!
    # define min max scaler
    scaler = MinMaxScaler()
    # transform data
    scaled = scaler.fit_transform([data]) # wrap array into 2d to avoid warning
    return scaled[0]

def get_macd_sessions(ticker_name, sell_today = False, period="max", data_all=None):
    if data_all is None:
        yticker = yf.Ticker(ticker_name)
        try:
            # df = yticker.history(period="2y") # max, 1y, 3mo
            df = yticker.history(period=period)  # max, 1y, 3mo
            #df = yticker.history(period='max')  # max, 1y, 3mo
        except:
            print('sth wrong with: ', ticker_name)
            return []
    else:
        if ticker_name not in data_all: return []
        if len(data_all[ticker_name]) == 0: return []
        df = data_all[ticker_name].copy()

    if len(df) == 0:
        return []

    time_range_start = df.index[0]

    df = get_MACD(df)
    buy_signals, sell_signals, signals = detect_macd_signals(df, sell_today=sell_today)
    rows_summary, total_profit, total_transaction = print_performance_summary(signals, time_range_start,
                                                                              PRINT_DETAIL=PRINT_DETAIL)
    macd_sessions = []

    for ri, rs in enumerate(rows_summary):
        hist_percentage = df['macd_hist_percentage'][:rs['buy_date'].strftime('%Y-%m-%d')]
        hist_percentage = hist_percentage.values
        macd_percentage = df['macd_percentage'][:rs['buy_date'].strftime('%Y-%m-%d')]
        macd_percentage = macd_percentage.values
        signal_percentage = df['macd_signal_percentage'][:rs['buy_date'].strftime('%Y-%m-%d')]
        signal_percentage = signal_percentage.values

        close = df['Close'][:rs['buy_date'].strftime('%Y-%m-%d')]
        close = close.values
        volume = df['Volume'][:rs['buy_date'].strftime('%Y-%m-%d')]
        volume = volume.values

        if len(hist_percentage) >= 120:  # if enough data
            # get past 120 days data only
            if min(close[-120:]) == max(close[-120:]) or min(volume[-120:]) == max(volume[-120:]):  # invalid data
                continue
            # if max(close[-120:]) < 1.0: #penny stock
            #    continue
            if np.isnan(np.sum(close[-120:])) or np.isnan(np.sum(volume[-120:])) \
                    or np.isnan(np.sum(hist_percentage[-120:])) or np.isnan(np.sum(macd_percentage[-120:])) or np.isnan(
                np.sum(signal_percentage[-120:])):
                continue
            scaled_close = simple_minmax_scaler(close[-120:])
            # scaled_close = array_MinMaxScaler(close[-120:])
            scaled_volume = simple_minmax_scaler(volume[-120:])
            # scaled_volume = array_MinMaxScaler(volume[-120:])
            dataset = np.vstack(
                (hist_percentage[-120:], macd_percentage[-120:], signal_percentage[-120:], scaled_close, scaled_volume))
            dataset = np.transpose(dataset)

            y = float(rs['profit_pct'].strip('%'))
            # x_train.append(dataset)
            # y_train.append(y) # floating value
            valid_each_row = rs
            valid_each_row['ticker'] = ticker_name
            valid_each_row['x'] = dataset
            # valid_each_row['close'] = simple_minmax_scaler(close[-120:])
            # valid_each_row['volume'] = simple_minmax_scaler(volume[-120:])
            valid_each_row['y'] = y

            macd_sessions.append(valid_each_row)
    return macd_sessions


#################################################################################################
PRINT_DETAIL = False


if __name__ == '__main__':
    # to generate training data, set it to False
    # to include latest MACD data, set to True
    sell_today = True

    date_now = time.strftime("%Y-%m-%d")

    """
    sp500tickers = si.tickers_sp500()
    nasdaqtickers = si.tickers_nasdaq()
    #dowtickers = si.tickers_dow()
    othertickers = si.tickers_other()
    alltickers = list(set(sp500tickers + nasdaqtickers + othertickers))
    mc = pd.read_csv('../backtesting_zipline/data/macd_better_than_10percent.csv')
    all_macd_better = mc['ticker'].tolist()
    """

    x_train, y_train, x_test, y_test = [], [], [], []
    total_try, total_valid = 0, 0
    valid_rows_summary = []

    #for ticker_name in  alltickers:
    for ticker_name in  sp500tickers:
    #for ticker_name in['MGM', 'NVDA']:
    #for ticker_name in all_macd_better:
        total_try += 1
        print(ticker_name)

        macd_sessions = get_macd_sessions(ticker_name, sell_today=sell_today)

        total_valid += 1
        valid_rows_summary.extend(macd_sessions)

    #convert x_train and y_train to numpy arrays
    #x_train, y_train , y_train2 = np.array(x_train), np.array(y_train), np.array(y_train2)
    #x_train, y_train2 = np.array(x_train), np.array(y_train2)

    #print(x_train.shape, y_train2.shape)
    print('total scan, total valid:', total_try, total_valid)

    #############################################################
    #pname = './data/all_ticker_last120x5_max.p'
    #pname = './data/sell_today_macd_better_last120x5_max_last_day_%s.p' % date_now
    pname = './data/sell_today_sp500_last120x5_max_last_day_%s.p' % date_now

    #pname = './data/test0.p'
    #pname = './data/test1.p'

    pickle.dump(valid_rows_summary, open(pname, 'wb'))
    print('datafile saved: ', pname)
    #############################################################

    #v2 = pickle.load(open(pname, 'rb'))
    #print('datafile load: ', pname)

    # double check last session
    valid_each_row = valid_rows_summary[-1]
    fig = plt.figure(0, figsize=(24, 40))
    #t0 = df.index[:rs['buy_date'].strftime('%Y-%m-%d')]
    #t1 = t0[-120:]
    t1=range(120)

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t1, valid_each_row['x'][:,3], marker='o')
    plt.title(valid_each_row['ticker'])

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(t1, valid_each_row['x'][:,4], marker='o')

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(t1, valid_each_row['x'][:,1:3], marker='o')
    plt.grid('on')

    plt.show()
