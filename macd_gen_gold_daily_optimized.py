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
import math
import numpy as np
from scipy.signal import lfilter, filtfilt
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, date, timedelta
import csv
import time
import pickle
import time

from macd_gen_data_nn import get_macd_sessions

############TODO: fix the import
import sys


pd.options.display.float_format = '{:.2f}%'.format

def get_gold_tickers(DAYS_CHECK_BACKWARD=0):
    if True: # get and save S&P500 tickers
        sp500tickers = si.tickers_sp500()
        if False:
            with open('./data/sp500_saved_tickers.txt', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(sp500tickers)
    else:
        with open('./data/sp500_saved_tickers.txt', newline='') as f:
            reader = csv.reader(f)
            sp500tickers = list(reader)[0]

    #DAYS_CHECK_BACKWARD = 0 #1 #7 #0 #7
    print(f'DAYS_CHECK_BACKWARD: {DAYS_CHECK_BACKWARD}')
    date_now = time.strftime("%Y-%m-%d")
    print(date_now)
    #date_now = "2021-04-27"
    trigger_date = datetime.strptime(date_now, '%Y-%m-%d') - timedelta(days = DAYS_CHECK_BACKWARD)
    print('trigger boundary:', trigger_date)

    s = time.time()

    target_list = sp500tickers[:]
    #target_list = ['EQIX', 'A']

    # download all data to save time
    data_all = yf.download(' '.join(target_list), period="max", group_by='tickers')

    ls_macd_recent_trigger = []
    valid_rows_summary = []
    for ticker_name in target_list:
    #for ticker_name in ['AIG', 'EQIX']:
        print(ticker_name)
        try:
            macd_sessions = get_macd_sessions(ticker_name, sell_today=True, period="max", data_all=data_all)
            if macd_sessions:
                latest_session = macd_sessions[-1]
                if latest_session['buy_date'] >= trigger_date:
                    print('trigger date:', latest_session['buy_date'])
                    ls_macd_recent_trigger.append(ticker_name)
                    valid_rows_summary.append(macd_sessions[-1])  # for inference, take the last one only
                # else:
                # print('trigger earlier:', latest_session['buy_date'])
        except:
            print('sth wrong with: ', ticker_name)


    print('MACD latest trigger: ', ls_macd_recent_trigger)
    with open('./data/nn_good_precision_latest_MACD_trigger.txt', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(ls_macd_recent_trigger)
    print('*'*80)
    print('file written: ', './data/nn_good_precision_latest_MACD_trigger.txt')

    pname = './data/sp500_tmp.p'
    pickle.dump(valid_rows_summary, open(pname, 'wb'))
    print('file written: ', './data/sp500_tmp.p')
    print('*'*80)

    e = time.time()
    print('----------------------------------------------------------------time: ', e - s)



if __name__ == '__main__':
    get_gold_tickers()