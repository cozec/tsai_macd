"""
functions for ensemble prediction

"""
import numpy as np
import pickle
from datetime import datetime, date, timedelta
from collections import defaultdict
import time
import pandas as pd
import math
import csv
import yfinance as yf

from parameters import *
pd.options.display.float_format = '{:.2f}%'.format

if False:
    import sys
    sys.path.insert(0, "../backtesting_zipline")
    from tutorial_trade_macd import detect_macd_signals, print_performance_summary
else:
    #from tutorial_trade_macd0 import detect_macd_signals, print_performance_summary
    pass


def p2f(x):
    return float(x.strip('%'))/100
# cannot handle constant input
def simple_minmax_scaler(X):
    mn, mx = X.min(), X.max()
    X_scaled = (X - mn) / (mx - mn)
    return X_scaled


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

def detect_macd_signals(data, sell_today = False):
    """Use MACD cross-over to decide buy/sell

    Args:
      data: panda DataFrame with OHLC with MACD data

    Return:
      buy_signals, sell_signals: for chart plot
      signals: buy/sell transaction for summary printing
    """

    buy_signals = [np.nan]
    sell_signals = [np.nan]
    signals = []
    last_signal = None

    for i in range(1, len(data)):
        if data['macd_hist'][i - 1] < 0 and data['macd_hist'][i] > 0:
            price = (data['Open'][i] + data['Close'][i]) / 2
            #price = data['Close'][i]
            buy_signals.append(price)
            last_signal = 'buy'
            signals.append({
                'date': data.index[i],
                'action': 'buy',
                'price': price
            })
            sell_signals.append(np.nan)
        elif data['macd_hist'][i - 1] > 0 and data['macd_hist'][i] < 0 and last_signal == 'buy':
            price = (data['Open'][i] + data['Close'][i]) / 2
            #price = data['Close'][i]
            sell_signals.append(price)
            last_signal = 'sell'
            signals.append({
                'date': data.index[i],
                'action': 'sell',
                'price': price
            })
            buy_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    if sell_today:
        # if last signal is 'buy', treat today as 'sell'
        today_price = (data['Open'][-1] + data['Close'][-1]) / 2
        #today_price = data['Close'][-1]
        if last_signal == 'buy':
            sell_signals.append(today_price)
            signals.append({
                'date': data.index[-1],
                'action': 'sell',
                'price': today_price
            })

    return buy_signals, sell_signals, signals


def print_performance_summary(signals, time_range_start, PRINT_DETAIL=True):
    """Print buy/sell transactions and statistics

    Args:
      signals: recorded buy/sell transactions
    """
    pairs = zip(*[iter(signals)] * 2)
    rows = []

    profit_count = 0
    profit_pct_avg = 1.0

    for (buy, sell) in pairs:
        if sell['date'] > time_range_start and buy['date'] > time_range_start:
            profit = sell['price'] - buy['price']
            profit_pct = profit / buy['price']

            if profit > 0:
                profit_count += 1
            profit_pct_avg *= 1 + profit_pct

            row = {
                'buy_date': buy['date'],
                'sell_date': sell['date'],
                'duration': (sell['date'] - buy['date']).days,
                #'profit': profit,
                'profit_pct': "{0:.2%}".format(profit_pct)
            }
            rows.append(row)

    df = pd.DataFrame(rows, columns=['buy_date', 'sell_date', 'duration', 'profit_pct'])
    if PRINT_DETAIL:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)

    total_transaction = len(rows) #math.floor(len(signals) / 2)
    stats = {
        #'total_transaction': total_transaction,
        #'profit_rate': "{0:.2%}".format(profit_count / total_transaction),
        #'avg_profit_per_transaction': "{0:.2%}".format(profit_pct_avg / total_transaction)
        'total_profit': "{0:.2%}".format(profit_pct_avg-1)
    }
    if PRINT_DETAIL:
        for key, value in stats.items():
            print('{0:30}  {1}'.format(key, value))
    return rows, profit_pct_avg-1, total_transaction

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
    rows_summary, total_profit, total_transaction = print_performance_summary(signals, time_range_start, PRINT_DETAIL=False)
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


def get_gold_tickers(DAYS_CHECK_BACKWARD=0):
    sp500tickers = si.tickers_sp500()

    #DAYS_CHECK_BACKWARD = 0 #1 #7 #0 #7
    print(f'DAYS_CHECK_BACKWARD: {DAYS_CHECK_BACKWARD}')
    date_now = time.strftime("%Y-%m-%d")
    print(date_now)
    #date_now = "2021-04-27"
    trigger_date = datetime.strptime(date_now, '%Y-%m-%d') - timedelta(days = DAYS_CHECK_BACKWARD)
    print('trigger boundary:', trigger_date)

    s = time.time()

    target_list = sp500tickers[:]
    #target_list = ['ABMD', 'CSCO', 'LOW']

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


def macd_get_train_test_simple(datafile, train_date_end,
                               train_date_start = datetime.strptime("1900-01-01", '%Y-%m-%d'),
                               BUY_WAIT_THRESHOLD = 2,
                               TRAIN_TEST_DATA_RANGE = 120,
                               modelType = 'dnn',
                               macd_only = False,
                               DEBUG = False,
                               INFERENCE_ONLY = False
                               ):
    print('loading pickle...')
    valid_rows_summary = pickle.load(open(datafile, 'rb'))

    # separate train, val, and test
    #train, val: all data before a date, say 02/01/2021
    #      test: all data after a date, say 02/01/2021
    x_all, y_all, x_test, y_test = [], [], [], []
    ls_test = []
    for rs in valid_rows_summary:
        profit = p2f(rs['profit_pct']) * 100
        #if profit < -50 or profit > 230:
        #    continue

        #if rs['buy_date'] < train_date_end:
        if train_date_start < rs['buy_date'] < train_date_end:
            #print('train:', rs['buy_date'])
            x_all.append(rs['x'])
            y_all.append(rs['y'])
        elif rs['buy_date'] >= train_date_end:
            #print('test:', rs['buy_date'])
            x_test.append(rs['x'])
            y_test.append(rs['y'])
            ls_test.append(rs)

    x_all, y_all = np.array(x_all), np.array(y_all)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # get y_all by different threshold; convert to sparse one below; seems redundant here
    #y_all = np.array([[1] if p > BUY_WAIT_THRESHOLD else [0] for p in y_all])
    #y_test = np.array([[1] if p > BUY_WAIT_THRESHOLD else [0] for p in y_test])

    y_all = np.array([1 if p > BUY_WAIT_THRESHOLD else 0 for p in y_all])
    y_test = np.array([1 if p > BUY_WAIT_THRESHOLD else 0 for p in y_test])

    if not INFERENCE_ONLY:
        # limit data range
        x_all = x_all[:, :TRAIN_TEST_DATA_RANGE, :]
        x_test = x_test[:, :TRAIN_TEST_DATA_RANGE, :]

    # debug using smaller set
    if DEBUG:
        print('debug mode.............................................')
        x_all = x_all[:20000]
        y_all = y_all[:20000]
        x_test, y_test = x_test[:2000], y_test[:2000]
        #x_test, y_test = x_test[:int(len(x_test)/10)], y_test[:int(len(y_test)/10)]

    if macd_only:
        # debug: only use 3 macd
        #x_all = x_all[:,:,:3]
        #x_test = x_test[:,:,:3]
        # debug: only use 1 macd
        x_all = x_all[:,:,4:5]
        x_test = x_test[:,:,4:5]

    if modelType == 'dnn':
        x_all = np.reshape(x_all, (x_all.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))


    if not INFERENCE_ONLY:
        x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=4, stratify=y_all)
        #x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, stratify=y_all)

        y_train = y_train.astype("float32")
        y_val = y_val.astype("float32")
    else:
        x_train, x_val, y_train, y_val = [],[],[],[]

    return x_train, x_val, x_test, y_train, y_val, y_test, ls_test

def ensemble_model_analysis(ensemble_pp, ensemble_preds_label, y_test, ls_test, test_range_start, test_range_end,
                            predict_save_path='nn_predictions',
                            pp_threshold=0.5,
                            INFERENCE_ONLY=False
                            ):

    all_op_after_model_date = []
    all_op_after_model_date_2_buys = []
    y_pred_list_tensor_tf_dnn = []
    all_buy_profit, nn_buy_profit = [], []
    dict_by_date = defaultdict(list)
    dict_test_label = defaultdict(list)
    dict_pred_label = defaultdict(list)
    dict_pred_label = defaultdict(list)

    # calculate monthly return
    #month_list = [i.strftime('%Y-%m-%d') for i in pd.date_range(start=test_range_start, end=test_range_end, freq='MS')]
    # always test 1/1/2020 to 5/1/2021

    ls_profit, ls_op_dates, ls_macd_original_profit = {}, {}, {}
    for m in test_month_list[:-1]:
        ls_profit[m] = []
        ls_op_dates[m] = []
        ls_macd_original_profit[m] = []

    ls_pp_macd_profit, ls_pp_temp = [], []
    for rs, predictions_tf_dnn, posterior in zip(ls_test, ensemble_preds_label, ensemble_pp):
        profit = p2f(rs['profit_pct']) * 100

        test_label = '-wait->' if p2f(rs['profit_pct']) * 100 < BUY_WAIT_THRESHOLD else '-buy->'
        test_value = 0 if p2f(rs['profit_pct']) * 100 < BUY_WAIT_THRESHOLD else 1
        all_buy_profit.append(p2f(rs['profit_pct']) * 100)

        buy_date = rs['buy_date']
        op_dates = abs((rs['sell_date'] - rs['buy_date']).days)
        for i, m in enumerate(test_month_list[:-1]):
            if datetime.strptime(test_month_list[i], '%Y-%m-%d') <= buy_date < datetime.strptime(test_month_list[i + 1], '%Y-%m-%d'):
                ls_macd_original_profit[m].append(profit)
                break

        #if predictions_tf_dnn == 0:
        #if posterior[0] < pp_threshold or posterior[1] < pp_threshold:
        # if any is 'w', ans is 'w'
        if (posterior < pp_threshold).any():
            pred_label_dnn = 'w'
            y_pred_list_tensor_tf_dnn.append(0)
        else:  # np.argmax(predictions) == 1:
            pred_label_dnn = 'b'
            y_pred_list_tensor_tf_dnn.append(1)
            nn_buy_profit.append(p2f(rs['profit_pct']) * 100)
            dict_by_date[rs['buy_date']].append(p2f(rs['profit_pct']) * 100)
            dict_pred_label[rs['buy_date']].append(1)
            dict_test_label[rs['buy_date']].append(test_value)

            for i, m in enumerate(test_month_list[:-1]):
                # to verify shorter trading window lead to worse profit
                #if datetime.strptime(test_month_list[i], '%Y-%m-%d') <= buy_date < datetime.strptime(test_month_list[i+1], '%Y-%m-%d') \
                #        and rs['sell_date'] < datetime.strptime(test_month_list[i+1], '%Y-%m-%d') + timedelta(days=4):
                if datetime.strptime(test_month_list[i], '%Y-%m-%d') <= buy_date < datetime.strptime(test_month_list[i+1], '%Y-%m-%d'):
                    ls_profit[m].append(profit)
                    ls_op_dates[m].append(op_dates)
                    break
        pred_label_dnns, number_of_buys = [], 0
        for p1 in posterior < pp_threshold:
            if p1:
                pred_label_dnns.append('w')
            else:
                pred_label_dnns.append('b')
                number_of_buys += 1
        pred_label_dnns = '_'.join(pred_label_dnns)
        combined_pp = '_'.join(['%.2f'%tt for tt in posterior])
        tt = {
            't': rs['ticker'],
            'p': p2f(rs['profit_pct']) * 100,
            't_label': test_label,
            'nn_p': pred_label_dnns,
            'posterior': combined_pp,
            'buy': rs['buy_date'].strftime('%Y-%m-%d'),
            'sell': rs['sell_date'].strftime('%Y-%m-%d'),
        }
        all_op_after_model_date.append(tt)
        if number_of_buys >= 1:
            all_op_after_model_date_2_buys.append(tt)

    # dict_by_date_sorted = sorted(dict_by_date)
    # print(sorted(dict_by_date.items(), key = lambda kv: (kv[1], kv[0])))
    # for i in sorted(dict_by_date):
    #    print((i, np.mean(dict_by_date[i])))
    profit_by_date = [[i, np.mean(dict_by_date[i])] for i in sorted(dict_by_date)]
    profit_by_date2 = [i for i in sorted(dict_by_date)]

    ###########################################################################################
    monthly_data = []
    for k, v in ls_profit.items():
        dd = {
            'month': k,
            test_range_start: np.mean(np.array(v)),
            #pp_threshold: np.mean(np.array(v)),
            'total op': len(v),
            'ave op dates': np.mean(np.array(ls_op_dates[k])),
            'org macd': np.mean(np.array(ls_macd_original_profit[k]))
        }
        monthly_data.append(dd)
    print('monthly data:')
    df_monthly_data = pd.DataFrame(monthly_data,)
    if not INFERENCE_ONLY:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
           print(df_monthly_data)
        df_monthly_data.to_csv(f'./{predict_save_path}/monthly_profits_%s.csv' % test_range_start)

    print('\nave of monthly profit: ', df_monthly_data[test_range_start].mean())

    test_by_date = [[i, dict_test_label[i]] for i in sorted(dict_test_label)]
    pred_by_date = [[i, dict_pred_label[i]] for i in sorted(dict_pred_label)]

    ls_precision_by_date = []
    for t0, p0 in zip(test_by_date, pred_by_date):
        t, p = np.array(t0[1]), np.array(p0[1])
        #cr_tf_dnn = classification_report(t, p, target_names=['wait', 'buy'], output_dict=True, zero_division=True)
        buy_precision = float(sum(t)) / len(t)
        ls_precision_by_date.append(buy_precision)
    ls_precision_by_date = np.array(ls_precision_by_date)


    print('all_op_after_model_date save to disk:')
    #df_temp = pd.DataFrame(all_op_after_model_date, columns=['t', 'p', 't_label', 'tf_dnn_p', 'buy', 'sell'])
    df_temp = pd.DataFrame(all_op_after_model_date)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_temp)
    df_temp.to_csv(f'./{predict_save_path}/3nn_after_model_date.csv')
    # df_temp.to_csv('./nn_predictions/3nn_precision_after_model_date_still_running.csv')


    print('*'*80)
    print('all_op_after_model_date_2_buys')
    if all_op_after_model_date_2_buys:
        df_temp = pd.DataFrame(all_op_after_model_date_2_buys)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_temp)
    else:
        print('no 2 buys')


    if not INFERENCE_ONLY:
        cr_tf_dnn = classification_report(y_test, y_pred_list_tensor_tf_dnn, target_names=['wait', 'buy'], output_dict=True)
        print('tf dnn buy precision: %.2f, tf dnn wait recall: %.2f' % (
            cr_tf_dnn['buy']['precision'], cr_tf_dnn['wait']['recall']))

        print(confusion_matrix(y_test, y_pred_list_tensor_tf_dnn))

        print('ave nn_buy_profit: ', np.mean(np.array(nn_buy_profit)))
        print('ave all_buy_profit: ', np.mean(np.array(all_buy_profit)))
        precision = cr_tf_dnn['buy']['precision']
        ratio = np.mean(np.array(nn_buy_profit)) / np.mean(np.array(all_buy_profit))
        print('pred win ratio: ', ratio)


    return df_monthly_data
def ensemble_trained_model_fnames():
    trained_models = []
    if True:
        model_path = 'LSTMPlus_good_models_2021-06-21'
        label = 'exp_12'
        fname_prefix = 'LSTMPlus-2021-06-21-CUTOFF-2020-01-01-BWT1-EP40'
        model_fname = f"model-{label}-" + fname_prefix
        learner_fname = f"learner-{label}-" + fname_prefix
        trained_models.append((model_path, model_fname, learner_fname, label, fname_prefix))
    if True:
        model_path = 'LSTMPlus_good_models_2021-06-21'
        label = 'exp_21'
        fname_prefix = 'LSTMPlus-2021-06-21-CUTOFF-2020-01-01-BWT1-EP40'
        model_fname = f"model-{label}-" + fname_prefix
        learner_fname = f"learner-{label}-" + fname_prefix
        trained_models.append((model_path, model_fname, learner_fname, label, fname_prefix))
    if True:
        model_path = 'LSTMPlus_good_models_2021-06-21'
        label = 'exp_15'
        fname_prefix = 'LSTMPlus-2021-06-23-CUTOFF-2020-01-01-BWT1-EP40-BATCH16'
        model_fname = f"model-{label}-" + fname_prefix
        learner_fname = f"learner-{label}-" + fname_prefix
        trained_models.append((model_path, model_fname, learner_fname, label, fname_prefix))
    if True:
        model_path = 'export'
        label = 'exp_4'
        fname_prefix = 'LSTMPlus-2021-06-27-CUTOFF-2020-01-01-BWT1-exp_4'
        model_fname = f"model-{fname_prefix}"
        learner_fname = f"learner-{fname_prefix}"
        trained_models.append((model_path, model_fname, learner_fname, label, fname_prefix))
    if True:
        model_path = 'LSTMPlus_predict_batch64_fix'
        label = 'exp_1'
        fname_prefix = 'LSTMPlus-2021-06-28-CUTOFF-2020-01-01-BWT1-EP20-exp_1'
        model_fname = f"model-{fname_prefix}"
        learner_fname = f"learner-{fname_prefix}"
        trained_models.append((model_path, model_fname, learner_fname, label, fname_prefix))
    if True:
        model_path = 'export_MLSTM_FCN_layers3'
        label = 'exp_7'
        fname_prefix = 'MLSTM_FCN_layers3-2021-06-24-CUTOFF-2020-01-01-BWT1-EP20'
        model_fname = f"model-{label}-" + fname_prefix
        learner_fname = f"learner-{label}-" + fname_prefix
        trained_models.append((model_path, model_fname, learner_fname, label, fname_prefix))
    if True:
        # wizard model
        model_path = 'best_models_tasi'
        label = 'exp_3'
        fname_prefix = 'LSTMPlus-2021-06-14-CUTOFF-2020-01-01-BWT1-EP40'
        model_fname = f"model-{label}-" + fname_prefix
        learner_fname = f"learner-{label}-" + fname_prefix
        trained_models.append((model_path, model_fname, learner_fname, label, fname_prefix))
    return trained_models
