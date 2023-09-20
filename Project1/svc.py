import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score

from preprocess_data import preprocess_credit_card_data, normal_train_test_split
from settings import BASE_DIR


def svc_initial_compare():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    # first without normalization
    st = time.time()
    svm_clf = SVC(kernel='linear', random_state=0, C=1.0, verbose=True)
    svm_clf.fit(X_train, y_train)

    train_time = time.time() - st

    y_pred = svm_clf.predict(X_test)

    out_dict = {'norm': {}, 'wo_norm': {}}
    out_dict['wo_norm']['svc_train_acc'] = svm_clf.score(X_train, y_train)
    out_dict['wo_norm']['svc_test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['wo_norm']['svc_test_recall'] = recall_score(y_test, y_pred)
    out_dict['wo_norm']['svc_test_f1'] = f1_score(y_test, y_pred)
    out_dict['wo_norm']['train_time'] = train_time

    # then with normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    st = time.time()

    svm_clf = SVC(kernel='linear', random_state=0, C=1.0, verbose=True)
    svm_clf.fit(X_train, y_train)

    train_time = time.time() - st

    y_pred = svm_clf.predict(X_test)

    out_dict['wo_norm']['svc_train_acc'] = svm_clf.score(X_train, y_train)
    out_dict['wo_norm']['svc_test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['wo_norm']['svc_test_recall'] = recall_score(y_test, y_pred)
    out_dict['wo_norm']['svc_test_f1'] = f1_score(y_test, y_pred)
    out_dict['wo_norm']['train_time'] = train_time

    return out_dict


def tune_svc_kernels():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    out_dict = {
        'kernel': [],
        'train_acc': [],
        'test_acc': [],
        'test_recall': [],
        'test_f1': [],
        'train_time': []
    }

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        print(kernel)
        st = time.time()
        svm_clf = SVC(kernel=kernel, random_state=0, C=1.0, verbose=True)
        svm_clf.fit(X_train, y_train)

        train_time = time.time() - st
        y_pred = svm_clf.predict(X_test)

        out_dict['kernel'].append(kernel)
        out_dict['train_acc'].append(svm_clf.score(X_train, y_train))
        out_dict['test_acc'].append(accuracy_score(y_test, y_pred))
        out_dict['test_recall'].append(recall_score(y_test, y_pred))
        out_dict['test_f1'].append(f1_score(y_test, y_pred))
        out_dict['train_time'].append(train_time)

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(os.path.join(BASE_DIR, 'results', 'svm_kernel_selection.csv'), index=False)

    return out_df


def plot_svc_tune_c(out_df):
    plt.plot(out_df['c'], out_df['train_acc'], label="train_accuracy")
    plt.plot(out_df['c'], out_df['test_acc'], label="test_accuracy")
    plt.legend()
    plt.title(f'Accuracy vs C for SVC (kernel = rbf)')
    plt.xlabel(f'C', fontsize=8)
    plt.ylabel('accuracy', fontsize=8)
    plt.savefig(os.path.join(BASE_DIR, 'plots', f'svc_c_tune.png'), dpi=300)
    plt.close()


def tune_svc_c():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    out_dict = {
        'c': [],
        'train_acc': [],
        'test_acc': [],
        'test_recall': [],
        'test_f1': [],
        'train_time': []
    }

    for c in np.arange(0.1, 1.0, 0.01):
        print(c)
        st = time.time()
        svm_clf = SVC(kernel='rbf', random_state=0, C=c, verbose=False)
        svm_clf.fit(X_train, y_train)

        train_time = time.time() - st
        y_pred = svm_clf.predict(X_test)

        out_dict['c'].append(c)
        out_dict['train_acc'].append(svm_clf.score(X_train, y_train))
        out_dict['test_acc'].append(accuracy_score(y_test, y_pred))
        out_dict['test_recall'].append(recall_score(y_test, y_pred))
        out_dict['test_f1'].append(f1_score(y_test, y_pred))
        out_dict['train_time'].append(train_time)

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(os.path.join(BASE_DIR, 'results', 'svm_tune_c.csv'), index=False)

    plot_svc_tune_c(out_df)

    return out_df


def run_svc():
    initial_compare_res = svc_initial_compare()
    print(initial_compare_res)
    tune_svc_kernels()
    tune_svc_c()
