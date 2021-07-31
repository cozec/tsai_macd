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

from sklearn.metrics import confusion_matrix, classification_report
from parameters import *
from batch_train_macd_tf_dnn import macd_get_train_test_simple, model_analysis, model_predict

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

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print('device--->', device)

    date_now = time.strftime("%Y-%m-%d")
    #pname = './data/sell_today_sp500_last120x5_max_last_day_2021-06-06.p'
    pname = './data/sell_today_sp500_last120x5_max_last_day_2021-06-28.p'

    predict_save_path = 'LSTMPlus_model_compare'
    if not os.path.isdir(predict_save_path):
        os.mkdir(f"./{predict_save_path}")

    #train_end = "2021-02-01"
    train_end = train_month_list[0]

    print('-' * 80, '\ntrain end: ', train_end)
    train_date_end = datetime.strptime(train_end, '%Y-%m-%d')
    print('BUY_WAIT_THRESHOLD: ', BUY_WAIT_THRESHOLD)

    x_train, x_val, x_test, y_train, y_valid, y_test, ls_test = macd_get_train_test_simple(pname, train_date_end,
                                                                                           BUY_WAIT_THRESHOLD=BUY_WAIT_THRESHOLD,
                                                                                           TRAIN_TEST_DATA_RANGE=TRAIN_TEST_DATA_RANGE,
                                                                                           modelType='rnn',
                                                                                           macd_only = False,
                                                                                           )

    # swap axes for tsai
    X_train = x_train.swapaxes(1, 2)
    X_valid = x_val.swapaxes(1, 2)
    X_test = x_test.swapaxes(1, 2)

    print('X_train: ', X_train.shape)
    print('X_valid: ', X_valid.shape)
    print('X_test: ', X_test.shape)
    unique, counts = np.unique(y_train, return_counts=True)
    print('train: wait, buy:', dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print('test: wait, buy:', dict(zip(unique, counts)))

    X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])

    pp_threshold = 0.5 #0.55
    #pp_threshold = 0.55

    trained_models = ensemble_trained_model_fnames()

    df_summary = pd.DataFrame()
    ensemble_pp, ensemble_preds_label = [], []
    #for model_path, model_fname, learner_fname, label in trained_models:
    for model_path, model_fname, learner_fname, label, fname_prefix in trained_models[5:] + trained_models[3:4]:
    #for model_path, model_fname, learner_fname, label in trained_models[5:] + trained_models[4:5]:
        print(label)
        print(model_fname)
        print(learner_fname)
        print('loading model...')

        learn = load_learner_all(path=model_path, dls_fname='dls', model_fname=model_fname, learner_fname=learner_fname)
        dls = learn.dls
        valid_dl = dls.valid

        # Labeled test data
        test_ds = valid_dl.dataset.add_test(X_test, y_test)  # In this case I'll use X and y, but this would be your test data
        test_dl = valid_dl.new(test_ds)
        # next(iter(test_dl))

        test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
        # test_probas, test_targets, test_preds
        print(f'accuracy: {skm.accuracy_score(test_targets, test_preds):10.6f}')

        preds_label = test_preds.cpu().detach().numpy()
        posterior = test_probas.cpu().detach().numpy()
        pp = posterior[:, 1]

        #df_monthly_data = model_analysis(pp, preds_label, y_test, ls_test, train_end, date_now)
        df_monthly_data, ave_of_monthly_profit = model_analysis(pp, preds_label, y_test, ls_test, train_end, date_now,
                                                                pp_threshold=pp_threshold,
                                                                BUY_WAIT_THRESHOLD=BUY_WAIT_THRESHOLD,
                                                                fname_prefix=fname_prefix + f"-{label}",
                                                                predict_save_path=predict_save_path)

        df_summary[label] = df_monthly_data[train_end]
        ensemble_pp.append(pp)
        ensemble_preds_label.append(preds_label)


    print(df_summary)
    df_summary.to_csv(f'./{predict_save_path}/summary_monthly.csv')

    ensemble_pp = np.array(ensemble_pp).T
    ensemble_preds_label = np.array(ensemble_preds_label).T
    df_monthly_data = ensemble_model_analysis(ensemble_pp, ensemble_preds_label, y_test, ls_test, train_end, date_now,
                                              predict_save_path=predict_save_path,
                                              pp_threshold=pp_threshold)
