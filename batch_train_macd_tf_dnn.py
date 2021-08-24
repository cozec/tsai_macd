"""
https://www.pluralsight.com/guides/deep-learning-model-perform-binary-classification

#Nvidia workshop sample code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu",
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))
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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from collections import defaultdict
#from tutorial_trade_macd import detect_macd_signals, print_performance_summary
#from zl_macd_basic_list import save_sp500_tickers

if False:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

from scipy.stats import binned_statistic

from parameters import *

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

def create_dnn_model0():
    model = keras.Sequential([
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        keras.layers.Dense(2, activation='softmax'),
    ])
    return model

def create_dnn_model_hidden_layer4():
    model = keras.Sequential([
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.2),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.2),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.3),
        keras.layers.Dense(2, activation='softmax'),
    ])
    return model

def create_dnn_model_hidden_layer7():
    model = keras.Sequential([
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.2),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.2),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.2),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.2),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.2),
        BatchNormalization(),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
        Dropout(0.3),
        keras.layers.Dense(2, activation='softmax'),
    ])
    return model


def model_predict(model_name, x_test, modelType = 'dnn'):
    # model_name = 'tf-dnn-2021-04-26-BWT2-EPOCHS15-SEQLEN600-CUTOFF-FEB'
    if modelType == 'dnn':
        model = keras.models.load_model('saved_models/dnn_model/' + model_name)
    else:
        model = keras.models.load_model('saved_models/rnn_model/' + model_name)
    if True:
        # load optimal model weights from results folder
        print('load model weights')
        if modelType == 'dnn':
            model_path = os.path.join("./saved_models/dnn_weights", model_name) + ".h5"
        else:
            model_path = os.path.join("./saved_models/rnn_weights", model_name) + ".h5"
        model.load_weights(model_path)

    #test_loss, test_acc = model.evaluate(x_test, y_test)
    #print('Test accuracy:', test_acc)
    predictions = model.predict(x_test)
    #print("predictions shape:", predictions.shape)
    preds_label = np.argmax(predictions, axis=1)
    return preds_label, predictions[:,1]

def model_analysis(pp, preds_label, y_test, ls_test, test_range_start, test_range_end, pp_threshold=0.5,
                   BUY_WAIT_THRESHOLD=1,
                   fname_prefix='',
                   predict_save_path = 'nn_predictions',
                   INFERENCE_ONLY=False
                   ):

    all_op_after_model_date = []
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
    for rs, predictions_tf_dnn, posterior in zip(ls_test, preds_label, pp):
        profit = p2f(rs['profit_pct']) * 100

        # remove macd profit outliers
        #if profit < 100:
        ls_pp_macd_profit.append([posterior, profit])
        if 0.9 > posterior > 0.8:
            ls_pp_temp.append(profit)
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
        if posterior < pp_threshold:
            pred_label_dnn = 'w'
            #pred_label_dnn = 'w(%.2f)' % posterior
            y_pred_list_tensor_tf_dnn.append(0)
        else:  # np.argmax(predictions) == 1:
            pred_label_dnn = 'b'
            #pred_label_dnn = 'b(%.2f)' % posterior


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

        tt = {
            't': rs['ticker'],
            'p': p2f(rs['profit_pct']) * 100,
            't_label': test_label,
            'posterior': '%.2f' % posterior,
            'tf_dnn_p': pred_label_dnn,
            'buy': rs['buy_date'].strftime('%Y-%m-%d'),
            'sell': rs['sell_date'].strftime('%Y-%m-%d'),
        }
        all_op_after_model_date.append(tt)

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
            pp_threshold: np.mean(np.array(v)),
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
        df_monthly_data.to_csv(f'./{predict_save_path}/monthly_profits_%s.csv' % fname_prefix)

    df_temp = pd.DataFrame(monthly_data,
                           columns=[test_range_start])

    if not INFERENCE_ONLY:
        print(df_temp[test_range_start].to_string(index=False))
        print('\nave of monthly profit: ', df_monthly_data[test_range_start].mean())

    #ls_month = df_monthly_data['month']

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
    df_temp = pd.DataFrame(all_op_after_model_date,
                           columns=['t', 'p', 't_label', 'posterior', 'tf_dnn_p', 'buy', 'sell'])
    if INFERENCE_ONLY:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
           print(df_temp)
    df_temp.to_csv(f'./{predict_save_path}/3nn_after_model_date_%s.csv' % fname_prefix)
    # df_temp.to_csv('./nn_predictions/3nn_precision_after_model_date_still_running.csv')

    if not INFERENCE_ONLY:
        cr_tf_dnn = classification_report(y_test, y_pred_list_tensor_tf_dnn, target_names=['wait', 'buy'], output_dict=True)
        print('tf dnn buy precision: %.2f, tf dnn wait recall: %.2f' % (
            cr_tf_dnn['buy']['precision'], cr_tf_dnn['wait']['recall']))

        print(confusion_matrix(y_test, y_pred_list_tensor_tf_dnn))

        #print('ave nn_buy_profit: ', np.mean(np.array(nn_buy_profit)))
        #print('ave all_buy_profit: ', np.mean(np.array(all_buy_profit)))
        precision = cr_tf_dnn['buy']['precision']
        ratio = np.mean(np.array(nn_buy_profit)) / np.mean(np.array(all_buy_profit))
        #print('pred win ratio: ', ratio)

    if False:
        fig = plt.figure(1, figsize=(18, 10))
        plt.clf()
        #ax = fig.add_subplot(111)
        dates = matplotlib.dates.date2num(profit_by_date2)
        # ax.set_xticks(dates)  # Tickmark + label at every plotted point
        # plt.plot(dates, np.array(profit_by_date)[:,1], marker='o')
        plt.plot_date(dates, np.array(profit_by_date)[:, 1], ls='-', marker='o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.xlabel('dates after training date end')
        plt.ylabel('ave profit')
        plt.grid(True)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        plt.savefig('plots/batch_' + model_name + '.png')

    if False:
        ls_pp_macd_profit = np.array(ls_pp_macd_profit)
        x_data = ls_pp_macd_profit[:, 0]
        y_data = ls_pp_macd_profit[:, 1]
        bin_means, bin_edges, misc = binned_statistic(x_data, y_data, statistic="mean", bins=10)

    #print('mean bin of 0.9:', np.mean(np.array(ls_pp_temp)))
    if False:
        fig = plt.figure(2, figsize=(18, 18))
        plt.clf()
        plt.subplot(2,1,1)
        #plt.scatter(ls_pp_macd_profit[:, 0], ls_pp_macd_profit[:, 1], marker='o')
        plt.plot(ls_pp_macd_profit[:, 0], ls_pp_macd_profit[:, 1], 'b.', label='raw data')
        plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='r', lw=10, label='binned statistic of data')
        plt.title('scatter plot: pp vs macd profit')
        plt.xlabel('pp')
        plt.ylabel('macd profit')
        plt.legend()

        plt.subplot(2,1,2)
        plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5, label='binned statistic of data')
        plt.title('macd profit vs. pp')
        plt.xlabel('pp')
        plt.ylabel('macd profit')

        plt.show()


    if False:
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        plt.plot_date(dates, ls_precision_by_date, ls='-', marker='o')
        plt.axhline(y=0.5, color='r', linestyle='-')
        plt.xlabel('dates after training date end')
        plt.ylabel('buy precison')
        plt.grid(True)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()

    if not INFERENCE_ONLY:
        ave_of_monthly_profit = df_monthly_data[test_range_start].mean()
    else:
        ave_of_monthly_profit = 0.0

    return df_monthly_data, ave_of_monthly_profit

if __name__ == '__main__':
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

    date_now = time.strftime("%Y-%m-%d")
    #date_now = "2021-04-27"

    HIDDEN_LAYER_NEURONS = 512
    BATCH_SIZE = 32 #128 #64

    #####################################
    TRAIN = False

    DEBUG = False
    if DEBUG:
        EPOCHS = 1 #40 #60 #15 #20
    else:
        EPOCHS = 40 #100  # 40 #60 #15 #20
    #####################################

    # create these folders if they does not exist
    if not os.path.isdir("saved_models/dnn_weights"):
        os.mkdir("./saved_models/dnn_weights")
    if not os.path.isdir("saved_models/dnn_model"):
        os.mkdir("./saved_models/dnn_model")

    #pname = './data/sell_today_macd_better_last120x5_max_last_day_2021-05-04.p'
    #pname = './data/sell_today_macd_better_last120x5_max_last_day_2021-05-18.p'
    #pname = './data/sell_today_macd_better_last120x5_max_last_day_2021-05-31.p'
    #pname = './data/macd_better_last120x5_max.p'
    #pname = './data/all_ticker_last120x5_max.p'

    pname = './data/sell_today_sp500_last120x5_max_last_day_2021-05-24.p'
    #pname = './data/sell_today_sp500_last120x5_max_last_day_2021-05-31.p'


    for train_end in train_month_list[:]:
        #train_end = "2020-01-01"
        print(train_end)
        #ls_ratio, ls_precisoin = [], []
        #for BUY_WAIT_THRESHOLD in range(6):
        train_date_end = datetime.strptime(train_end, '%Y-%m-%d')

        x_train, x_val, x_test, y_train, y_val, y_test, ls_test = macd_get_train_test_simple(
            pname, train_date_end, BUY_WAIT_THRESHOLD=BUY_WAIT_THRESHOLD, modelType='dnn', DEBUG=DEBUG, macd_only=False)

        print('train and val:', x_train.shape, y_train.shape, x_val.shape, y_val.shape)
        print('test: ', x_test.shape, y_test.shape)

        unique, counts = np.unique(y_train, return_counts=True)
        print('train: wait, buy:', dict(zip(unique, counts)))
        unique, counts = np.unique(y_test, return_counts=True)
        print('test: wait, buy:', dict(zip(unique, counts)))

        model_name = f"tf-dnn-{date_now}-BWT{BUY_WAIT_THRESHOLD}-EPOCHS{EPOCHS}-SEQLEN{x_train.shape[1]}-CUTOFF-{train_end}"
        #model_name = f"tf-dnn-BWT{BUY_WAIT_THRESHOLD}-EPOCHS{EPOCHS}-SEQLEN{x_train.shape[1]}-CUTOFF-{train_end}-MACD-ONLY"

        if TRAIN:  # train
            print('-' * 80, '\ntrain end: ', train_end)
            print('-' * 80, '\nBUY_WAIT_THRESHOLD: ', BUY_WAIT_THRESHOLD)


            print(model_name)
            model = create_dnn_model_hidden_layer7()
            model.compile(optimizer='adam',
                          loss='SparseCategoricalCrossentropy',
                          metrics=['SparseCategoricalAccuracy'])

            # some tensorflow callbacks
            checkpointer = ModelCheckpoint(os.path.join("saved_models/dnn_weights", model_name + ".h5"),
                                           save_weights_only=True,
                                           save_best_only=True,
                                           monitor='val_sparse_categorical_accuracy',
                                           verbose=1)

            history = model.fit(x_train, y_train,
                                epochs=EPOCHS,
                                batch_size = BATCH_SIZE,
                                #class_weight=class_weight,
                                callbacks=checkpointer,
                                validation_data=(x_val, y_val))

            print('-----------testing--------------------------------------------------------')
            test_loss, test_acc = model.evaluate(x_test, y_test)
            print('Test accuracy:', test_acc)


            model.save('saved_models/dnn_model/' + model_name)
            print('model saved: ', 'saved_models/dnn_model/' + model_name)

            # Plot history
            fig = plt.figure(0)
            plt.clf()
            plt.plot(history.history['sparse_categorical_accuracy'], label='Accuracy (training data)')
            plt.plot(history.history['val_sparse_categorical_accuracy'], label='Accuracy (validation data)')
            plt.ylabel('Accu value')
            plt.xlabel('No. epoch')
            plt.legend(loc="upper left")
            plt.savefig('plots/batch_train_' + model_name + '.png')
            #print('training plot saved as: ', 'saved_models/dnn_model/' + model_name + '.png')
            #plt.show()

            preds_label, pp  = model_predict(model_name, x_test, modelType = 'dnn')
            df_monthly_data = model_analysis(pp, preds_label, y_test, ls_test, train_end, date_now)

            if train_end == test_month_list[0]: # "2020-01-01":
                df_month_summary = df_monthly_data[['month', train_end]]
            else:
                df_month_summary = df_month_summary.join(df_monthly_data[train_end])



        else: # test
            #model_name = 'tf-dnn-BWT1-EPOCHS100-SEQLEN600-CUTOFF-2020-11-01-MACD-ONLY'
            #model_name = 'tf-dnn-2021-05-14-BWT1-EPOCHS100-SEQLEN600-CUTOFF-2018-12-01' #best model 0
            #model_name = 'tf-dnn-BWT1-EPOCHS100-SEQLEN600-CUTOFF-2020-01-01-MACD-ONLY' #best model 1
            #model_name = 'tf-dnn-2021-05-17-BWT1-EPOCHS100-SEQLEN600-CUTOFF-2020-01-01' #all ticker model
            model_name = 'tf-dnn-2021-05-24-BWT1-EPOCHS40-SEQLEN600-CUTOFF-2020-01-01' #sp500 base model

            model_path = '/home/ywang/stocks/macd_nn/saved_models_best/dnn_model/'
            print('model name: ', model_name)

            preds_label, pp = model_predict(model_name, x_test, modelType = 'dnn')

            ls_thresholds = [0.5] #, 0.9] # [0.5, 0.6, 0.7, 0.8, 0.9]
            for pp_threshold in ls_thresholds:
                df_monthly_data, _ = model_analysis(pp, preds_label, y_test, ls_test, train_end, date_now, pp_threshold=pp_threshold)
                if train_end == test_month_list[0]:
                    df_month_summary = df_monthly_data[['month', train_end]]
                else:
                    df_month_summary = df_month_summary.join(df_monthly_data[train_end])
                #print(df_month_summary)

                if False:
                    if pp_threshold == ls_thresholds[0]:
                        df_month_th = df_monthly_data[['month', pp_threshold]]
                    else:
                        df_month_th = df_month_th.join(df_monthly_data[pp_threshold])

                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(df_month_th)

            if False:
                fig = plt.figure(3, figsize=(18, 10))
                plt.clf()
                ax = plt.gca()
                for pp_threshold in ls_thresholds:
                    df_month_th.plot(kind='line', x='month', y=pp_threshold, ax=ax)
                    #df_month_th.plot(kind='line', x='month', y=0.6, ax=ax) #, color='red')

                plt.show()

    #print(df_month_summary)
    #df_month_summary.to_csv('./nn_predictions/summary_monthly_profits.csv')
