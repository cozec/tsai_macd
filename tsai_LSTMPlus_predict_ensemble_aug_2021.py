"""
3 best models trained with TSAI, the last model is the 'wizard' model, which give highest buy precision
Step 1: Get MACD gold trigger tickers and save the data
Step 2: Inference by 3 models

https://timeseriesai.github.io/tsai/tutorials.html
https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb#scrollTo=x1j4RDRB2WjF

Author: Adam Wang

Created: 20, August, 2021
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, date, timedelta
import tsai
from tsai.all import *

#from sklearn.metrics import confusion_matrix, classification_report

from parameters import *

#from batch_train_macd_tf_dnn import macd_get_train_test_simple, model_analysis, model_predict
#from macd_gen_data_nn import get_macd_sessions
#from tsai_models_ensemble import ensemble_trained_model_fnames, ensemble_model_analysis
#from macd_gen_gold_daily_optimized import get_gold_tickers

from tsai_predict_funs import macd_get_train_test_simple, p2f, ensemble_trained_model_fnames, ensemble_model_analysis, get_gold_tickers
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print('device--->', device)

    predict_save_path = 'LSTMPlus_predict_ensemble'
    if not os.path.isdir(predict_save_path):
        os.mkdir(f"./{predict_save_path}")

    date_now = time.strftime("%Y-%m-%d")
    train_end = train_month_list[0]
    pp_threshold = 0.5  # 0.55

    get_gold_tickers(DAYS_CHECK_BACKWARD=2)


    pname = './data/sp500_tmp.p'
    x_train, x_val, x_test, y_train, y_valid, y_test, ls_test = macd_get_train_test_simple(pname,
                                                                                           datetime.strptime("1900-01-01", '%Y-%m-%d'), #make sure all data into test
                                                                                           BUY_WAIT_THRESHOLD=BUY_WAIT_THRESHOLD,
                                                                                           modelType='rnn',
                                                                                           INFERENCE_ONLY=True
                                                                                           )
    X_test = x_test.swapaxes(1, 2)

    ensemble_pp, ensemble_preds_label = [], []
    trained_models = ensemble_trained_model_fnames()
    for model_path, model_fname, learner_fname, label, fname_prefix in  trained_models[3:4] + trained_models[5:]: # include wizard model
        print('loading model...', model_fname)
        learn = load_learner_all(path=model_path, dls_fname='dls', model_fname=model_fname, learner_fname=learner_fname)
        dls = learn.dls
        valid_dl = dls.valid

        test_ds = valid_dl.dataset.add_test(X_test, y_test)  # In this case I'll use X and y, but this would be your test data
        test_dl = valid_dl.new(test_ds)

        test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)

        preds_label = test_preds.cpu().detach().numpy()
        posterior = test_probas.cpu().detach().numpy()
        pp = posterior[:, 1]

        ensemble_pp.append(pp)
        ensemble_preds_label.append(preds_label)

    ensemble_pp = np.array(ensemble_pp).T
    ensemble_preds_label = np.array(ensemble_preds_label).T
    _ = ensemble_model_analysis(ensemble_pp, ensemble_preds_label, y_test, ls_test, train_end, date_now,
                                              predict_save_path=predict_save_path,
                                              INFERENCE_ONLY=True,
                                              pp_threshold=pp_threshold)
