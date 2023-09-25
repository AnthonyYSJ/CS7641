import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

from preprocess_data import preprocess_credit_card_data, smote_train_test_split
from settings import BASE_DIR


def train_knn(knn_clf, X_train, X_test, y_train, y_test):
    st = time.time()
    knn_clf.fit(X_train, y_train)
    train_time = time.time() - st

    y_pred = knn_clf.predict(X_test)
    y_pred_train = knn_clf.predict(X_train)

    out_dict = {}
    out_dict['train_acc'] = accuracy_score(y_train, y_pred_train)
    out_dict['train_recall'] = recall_score(y_train, y_pred_train)
    out_dict['train_f1'] = f1_score(y_train, y_pred_train)
    out_dict['test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['test_recall'] = recall_score(y_test, y_pred)
    out_dict['test_f1'] = f1_score(y_test, y_pred)
    out_dict['train_time'] = train_time

    return out_dict


def knn_initial_no_tune():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)

    knn_clf = KNeighborsClassifier()

    res = train_knn(knn_clf, X_train, X_test, y_train, y_test)

    return res


def plot_knn_n_neighbors(out_dict):
    assert len(set(out_dict['param_type'])) == 1
    param_type = list(set(out_dict['param_type']))[0]
    fig, ax = plt.subplots(3, figsize=(10, 15))
    ax[0].plot(out_dict['param_val'], out_dict['train_acc'], label='train_acc')
    ax[0].plot(out_dict['param_val'], out_dict['test_acc'], label='test_acc')
    ax[0].set_ylabel('accuracy', fontsize=10)
    ax[0].set_xlabel(f'{param_type}', fontsize=10)
    ax[0].legend()
    ax[0].set_title(f'knn accuracy vs {param_type}')

    ax[1].plot(out_dict['param_val'], out_dict['train_recall'], label='train_recall')
    ax[1].plot(out_dict['param_val'], out_dict['test_recall'], label='test_recall')
    ax[1].set_ylabel('recall', fontsize=10)
    ax[1].set_xlabel(f'{param_type}', fontsize=10)
    ax[1].legend()
    ax[1].set_title(f'knn recall vs {param_type}')

    ax[2].plot(out_dict['param_val'], out_dict['train_time'], label='train_time')
    ax[2].set_xlabel(f'{param_type}', fontsize=10)
    ax[2].legend()
    ax[2].set_title(f'knn train_time vs {param_type}')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'knn_scores_{param_type}.png'))
    plt.close()


def tune_knn(param_type):
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)

    out_dict = {
        'param_type': [],
        'param_val': [],
        'train_acc': [],
        'train_recall': [],
        'train_f1': [],
        'test_acc': [],
        'test_recall': [],
        'test_f1': [],
        'train_time': [],
    }

    if param_type == 'n_neighbors':
        for i in np.arange(5, 35, 5):
            print(f'param_type = {param_type}, param_val = {i}')
            knn_clf = KNeighborsClassifier(n_neighbors=i)
            res = train_knn(knn_clf, X_train, X_test, y_train, y_test)
            out_dict['param_type'].append(param_type)
            out_dict['param_val'].append(i)
            out_dict['train_acc'].append(res['train_acc'])
            out_dict['train_recall'].append(res['train_recall'])
            out_dict['train_f1'].append(res['train_f1'])
            out_dict['test_acc'].append(res['test_acc'])
            out_dict['test_recall'].append(res['test_recall'])
            out_dict['test_f1'].append(res['test_f1'])
            out_dict['train_time'].append(res['train_time'])
        plot_knn_n_neighbors(out_dict)
    elif param_type == 'weights':
        for i in ['uniform', 'distance']:
            print(f'param_type = {param_type}, param_val = {i}')
            knn_clf = KNeighborsClassifier(weights=i)
            res = train_knn(knn_clf, X_train, X_test, y_train, y_test)
            out_dict['param_type'].append(param_type)
            out_dict['param_val'].append(i)
            out_dict['train_acc'].append(res['train_acc'])
            out_dict['train_recall'].append(res['train_recall'])
            out_dict['train_f1'].append(res['train_f1'])
            out_dict['test_acc'].append(res['test_acc'])
            out_dict['test_recall'].append(res['test_recall'])
            out_dict['test_f1'].append(res['test_f1'])
            out_dict['train_time'].append(res['train_time'])
    else:  # param_type == 'metric'
        for i in ['minkowski', 'euclidean', 'manhattan']:
            print(f'param_type = {param_type}, param_val = {i}')
            knn_clf = KNeighborsClassifier(metric=i)
            res = train_knn(knn_clf, X_train, X_test, y_train, y_test)
            out_dict['param_type'].append(param_type)
            out_dict['param_val'].append(i)
            out_dict['train_acc'].append(res['train_acc'])
            out_dict['train_recall'].append(res['train_recall'])
            out_dict['train_f1'].append(res['train_f1'])
            out_dict['test_acc'].append(res['test_acc'])
            out_dict['test_recall'].append(res['test_recall'])
            out_dict['test_f1'].append(res['test_f1'])
            out_dict['train_time'].append(res['train_time'])

    df = pd.DataFrame(out_dict)
    df.to_csv(os.path.join(BASE_DIR, 'results', f'knn_tune_{param_type}.csv'), index=False)


def knn_best():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)

    knn_clf_best = KNeighborsClassifier(n_neighbors=25, weights='uniform', metric='manhattan')

    best_res = train_knn(knn_clf_best, X_train, X_test, y_train, y_test)

    return best_res


def run_knn():
    res_no_tune = knn_initial_no_tune()
    print(res_no_tune)

    tune_knn('n_neighbors')
    tune_knn('weights')
    tune_knn('metric')

    best_knn_res = knn_best()
    print(best_knn_res)
