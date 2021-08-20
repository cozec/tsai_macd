"""
https://timeseriesai.github.io/tsai/tutorials.html
https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb#scrollTo=x1j4RDRB2WjF

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, date, timedelta

import tsai
from tsai.all import *

if True:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print('device--->', device)

from sklearn.metrics import confusion_matrix, classification_report

from parameters import *
#TRAIN_TEST_DATA_RANGE = 105 #120 #60

from batch_train_macd_tf_dnn import macd_get_train_test_simple, model_analysis, model_predict
from macd_gen_data_nn import get_macd_sessions
from tsai_models_ensemble import ensemble_trained_model_fnames, ensemble_model_analysis
from macd_gen_gold_daily_optimized import get_gold_tickers

if __name__ == '__main__':

    get_gold_tickers(DAYS_CHECK_BACKWARD=7)

    predict_save_path = 'LSTMPlus_predict_ensemble'
    if not os.path.isdir(predict_save_path):
        os.mkdir(f"./{predict_save_path}")

    date_now = time.strftime("%Y-%m-%d")
    train_end = train_month_list[0]
    pp_threshold = 0.5  # 0.55

    #######################################################################################################
    pname = './data/sp500_tmp.p'
    if False:
        #inference given target tickers
        print('*'*80)
        with open('./data/nn_good_precision_latest_MACD_trigger.txt', newline='') as f:
            reader = csv.reader(f)
            MACD_latest_trigger = list(reader)[0]

        valid_rows_summary = []
        # move this part to "macd_gen_gold_list_for_nn_test.py" to save some time
        print('get_macd_sessions... (TODO: move to macd_gen_gold_list_for_nn_test.py)')
        for ticker_name in MACD_latest_trigger[:]:
        #for ticker_name in ['VIAC']:
            macd_sessions = get_macd_sessions(ticker_name, sell_today=True) #, period="2y")
            valid_rows_summary.append(macd_sessions[-1]) # for inference, take the last one only
            #valid_rows_summary.append(macd_sessions[-2]) # for inference, take the last one only

        pickle.dump(valid_rows_summary, open(pname, 'wb'))
    else:
        print('pickle already saved by macd_gen_gold_list_for_nn_test.py')

    x_train, x_val, x_test, y_train, y_valid, y_test, ls_test = macd_get_train_test_simple(pname,
                                                                                           datetime.strptime("1900-01-01", '%Y-%m-%d'), #make sure all data into test
                                                                                           BUY_WAIT_THRESHOLD=BUY_WAIT_THRESHOLD,
                                                                                           modelType='rnn',
                                                                                           INFERENCE_ONLY=True
                                                                                           )
    #print('train and val:', x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    print('test: ', x_test.shape, y_test.shape)
    X_test = x_test.swapaxes(1, 2)

    ensemble_pp, ensemble_preds_label = [], []
    trained_models = ensemble_trained_model_fnames()
    for model_path, model_fname, learner_fname, label, fname_prefix in  trained_models[3:4] + trained_models[5:]: # include wizard model
    #for model_path, model_fname, learner_fname, label, fname_prefix in  trained_models[3:4] + trained_models[5:6]:
        #model_path, model_fname, learner_fname, label, fname_prefix = trained_models[5]

        print(model_fname)
        #print(learner_fname)

        print('loading model...')
        learn = load_learner_all(path=model_path, dls_fname='dls', model_fname=model_fname, learner_fname=learner_fname)
        dls = learn.dls
        valid_dl = dls.valid

        test_ds = valid_dl.dataset.add_test(X_test, y_test)  # In this case I'll use X and y, but this would be your test data
        test_dl = valid_dl.new(test_ds)
        # next(iter(test_dl))

        test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
        # test_probas, test_targets, test_preds
        #print('test_targets, test_preds shape:', test_targets.shape, test_preds.shape)
        #print(f'accuracy: {skm.accuracy_score(test_targets, test_preds):10.6f}')

        preds_label = test_preds.cpu().detach().numpy()
        #print('predictions shape:', preds_label.shape)
        posterior = test_probas.cpu().detach().numpy()
        pp = posterior[:, 1]

        if False:
            #df_monthly_data = model_analysis(pp, preds_label, y_test, ls_test, train_end, date_now)
            df_monthly_data, ave_of_monthly_profit = model_analysis(pp, preds_label, y_test, ls_test, train_end, date_now,
                                                                    pp_threshold=pp_threshold,
                                                                    BUY_WAIT_THRESHOLD=BUY_WAIT_THRESHOLD,
                                                                    fname_prefix='predict-' + fname_prefix + f"-{label}",
                                                                    predict_save_path = predict_save_path,
                                                                    INFERENCE_ONLY=True
                                                                    )
        ensemble_pp.append(pp)
        ensemble_preds_label.append(preds_label)

    ensemble_pp = np.array(ensemble_pp).T
    ensemble_preds_label = np.array(ensemble_preds_label).T
    _ = ensemble_model_analysis(ensemble_pp, ensemble_preds_label, y_test, ls_test, train_end, date_now,
                                              predict_save_path=predict_save_path,
                                              INFERENCE_ONLY=True,
                                              pp_threshold=pp_threshold)
