import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from preprocess_data import preprocess_credit_card_data, normal_train_test_split
from settings import BASE_DIR


def train_knn(knn_clf, X_train, X_test, y_train, y_test):
    st = time.time()
    knn_clf.fit(X_train, y_train)

    train_time = time.time() - st

    y_pred = knn_clf.predict(X_test)
    out_dict = {}
    out_dict['train_acc'] = knn_clf.score(X_train, y_train)
    out_dict['test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['test_recall'] = recall_score(y_test, y_pred)
    out_dict['test_f1'] = f1_score(y_test, y_pred)
    out_dict['train_time'] = train_time

    return out_dict


def get_knn_res_comparing_norm():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    knn_clf = KNeighborsClassifier()

    res = train_knn(knn_clf, X_train, X_test, y_train, y_test)

    out_dict = {'wo_norm': {}, 'norm': {}}
    out_dict['wo_norm']['train_acc'] = res['train_acc']
    out_dict['wo_norm']['test_acc'] = res['test_acc']
    out_dict['wo_norm']['test_recall'] = res['test_recall']
    out_dict['wo_norm']['test_f1'] = res['test_f1']
    out_dict['wo_norm']['train_time'] = res['train_time']

    # then with normalization
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.fit_transform(X_test)

    knn_clf = KNeighborsClassifier()

    res = train_knn(knn_clf, X_train_norm, X_test_norm, y_train, y_test)

    out_dict['norm']['train_acc'] = res['train_acc']
    out_dict['norm']['test_acc'] = res['test_acc']
    out_dict['norm']['test_recall'] = res['test_recall']
    out_dict['norm']['test_f1'] = res['test_f1']
    out_dict['norm']['train_time'] = res['train_time']

    return out_dict


def plot_knn_n_neighbors(out_dict):
    fig, ax = plt.subplots(2, figsize=(9, 12))
    ax[0].plot(out_dict['param_val'], out_dict['train_acc'], label='train_acc')
    ax[0].plot(out_dict['param_val'], out_dict['test_acc'], label='test_acc')
    ax[0].set_ylabel('accuracy', fontsize=10)
    ax[0].set_xlabel('n_neighbors', fontsize=10)
    ax[0].legend()
    ax[0].set_title('knn accuracy vs n_neighbors')

    ax[1].plot(out_dict['param_val'], out_dict['train_time'], label='train_time')
    ax[1].set_xlabel('n_neighbors', fontsize=10)
    ax[1].legend()
    ax[1].set_title('knn train_time vs n_neighbors')

    fig.savefig(os.path.join(BASE_DIR, 'plots', 'knn_accuracy_n_neighbors.png'))


def tune_knn(param_type):
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.fit_transform(X_test)

    out_dict = {
        'param_type': [],
        'param_val': [],
        'train_acc': [],
        'test_acc': [],
        'test_recall': [],
        'test_f1': [],
        'train_time': [],
    }

    if param_type == 'n_neighbors':
        for i in np.arange(1, 100, 1):
            knn_clf = KNeighborsClassifier(n_neighbors=i)
            res = train_knn(knn_clf, X_train_norm, X_test_norm, y_train, y_test)
            out_dict['param_type'].append(param_type)
            out_dict['param_val'].append(i)
            out_dict['train_acc'].append(res['train_acc'])
            out_dict['test_acc'].append(res['test_acc'])
            out_dict['test_recall'].append(res['test_recall'])
            out_dict['test_f1'].append(res['test_f1'])
            out_dict['train_time'].append(res['train_time'])
        plot_knn_n_neighbors(out_dict)
    elif param_type == 'weights':
        for i in ['uniform', 'distance']:
            knn_clf = KNeighborsClassifier(weights=i)
            res = train_knn(knn_clf, X_train_norm, X_test_norm, y_train, y_test)
            out_dict['param_type'].append(param_type)
            out_dict['param_val'].append(i)
            out_dict['train_acc'].append(res['train_acc'])
            out_dict['test_acc'].append(res['test_acc'])
            out_dict['test_recall'].append(res['test_recall'])
            out_dict['test_f1'].append(res['test_f1'])
            out_dict['train_time'].append(res['train_time'])
    else:  # param_type == 'metric'
        for i in ['minkowski', 'euclidean', 'manhattan']:
            knn_clf = KNeighborsClassifier(metric=i)
            res = train_knn(knn_clf, X_train_norm, X_test_norm, y_train, y_test)
            out_dict['param_type'].append(param_type)
            out_dict['param_val'].append(i)
            out_dict['train_acc'].append(res['train_acc'])
            out_dict['test_acc'].append(res['test_acc'])
            out_dict['test_recall'].append(res['test_recall'])
            out_dict['test_f1'].append(res['test_f1'])
            out_dict['train_time'].append(res['train_time'])

    df = pd.DataFrame(out_dict)
    df.to_csv(os.path.join(BASE_DIR, 'results', f'knn_tune_{param_type}.csv'), index=False)


def knn_best():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.fit_transform(X_test)

    grid_params = {
        'n_neighbors': [i for i in np.arange(1, 21, 1)],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'euclidean', 'manhattan'],
    }
    knn_gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=5, n_jobs=1)
    knn_gs_res = knn_gs.fit(X_train_norm, y_train)

    print(knn_gs_res.best_params_)

    knn_clf_best = KNeighborsClassifier(n_neighbors=6, weights='uniform', metric='manhattan')

    best_res = train_knn(knn_clf_best, X_train_norm, X_test_norm, y_train, y_test)
    print(best_res)


def run_knn():
    knn_res_comparing_norm = get_knn_res_comparing_norm()

    tune_knn('n_neighbors')
    tune_knn('weights')
    tune_knn('metric')

    knn_best()
