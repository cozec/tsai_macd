from yahoo_fin import stock_info as si
import pandas as pd
from datetime import datetime, date, timedelta

BUY_WAIT_THRESHOLD = 1 #0 #1 #2 #percent
TRAIN_TEST_DATA_RANGE = 120 #60

sp500tickers = si.tickers_sp500()
nasdaqtickers = si.tickers_nasdaq()
#dowtickers = si.tickers_dow()
othertickers = si.tickers_other()
alltickers = list(set(sp500tickers + nasdaqtickers + othertickers))
mc = pd.read_csv('../backtesting_zipline/data/macd_better_than_10percent.csv')
all_macd_better = mc['ticker'].tolist()

#train_month_list = [i.strftime('%Y-%m-%d') for i in pd.date_range(start="2018-11-01", end="2019-12-01", freq='MS')]
#train_month_list = ["2018-12-01"]
#train_month_list = ["2020-01-01"]
train_month_list = [i.strftime('%Y-%m-%d') for i in pd.date_range(start="2020-01-01", end="2021-05-01", freq='MS')]

#test_month_list = [i.strftime('%Y-%m-%d') for i in pd.date_range(start="2020-01-01", end="2021-05-01", freq='MS')]
#test_month_list = [i.strftime('%Y-%m-%d') for i in pd.date_range(start="2020-06-01", end="2021-06-01", freq='MS')]
test_month_list = [i.strftime('%Y-%m-%d') for i in pd.date_range(start="2020-06-01", end="2021-07-01", freq='MS')]

#test_month_list = [i.strftime('%Y-%m-%d') for i in pd.date_range(start=train_month_list[0], end="2021-05-01", freq='MS')]
#test_month_list = [i.strftime('%Y-%m-%d') for i in pd.date_range(start=train_month_list[0], end="2021-06-01", freq='MS')]

#print('train_month_list', train_month_list)
#print('test_month_list', test_month_list)

#BUY_PRECISION_TH = 0.67 #0.9 # 0.67 #0.5
may_01_2021_date = datetime.strptime("2021-05-01", '%Y-%m-%d')


#pname = './data/macd_better_last120x5_max.p'

def p2f(x):
    return float(x.strip('%'))/100


