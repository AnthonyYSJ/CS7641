import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score

from preprocess_data import preprocess_credit_card_data, smote_train_test_split
from settings import BASE_DIR


def svc_initial_compare():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)
    X_train = X_train[0:int(len(X_train) * 0.1)]
    y_train = y_train[0:int(len(y_train) * 0.1)]

    svm_clf = SVC(kernel='linear', random_state=0, C=1.0, verbose=False)

    st = time.time()
    svm_clf.fit(X_train, y_train)
    train_time = time.time() - st

    y_pred = svm_clf.predict(X_test)
    y_pred_train = svm_clf.predict(X_train)

    out_dict = {}
    out_dict['train_acc'] = accuracy_score(y_train, y_pred_train)
    out_dict['train_recall'] = recall_score(y_train, y_pred_train)
    out_dict['train_f1'] = f1_score(y_train, y_pred_train)
    out_dict['test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['test_recall'] = recall_score(y_test, y_pred)
    out_dict['test_f1'] = f1_score(y_test, y_pred)
    out_dict['train_time'] = train_time

    return out_dict


def tune_svc_kernels():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)
    X_train = X_train[0:int(len(X_train) * 0.2)]
    y_train = y_train[0:int(len(y_train) * 0.2)]

    out_dict = {
        'kernel': [],
        'train_acc': [],
        'train_recall': [],
        'train_f1': [],
        'test_acc': [],
        'test_recall': [],
        'test_f1': [],
        'train_time': [],
    }

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        print(kernel)
        st = time.time()
        svm_clf = SVC(kernel=kernel, random_state=0, C=1.0, verbose=True)
        svm_clf.fit(X_train, y_train)

        train_time = time.time() - st
        y_pred = svm_clf.predict(X_test)
        y_pred_train = svm_clf.predict(X_train)

        out_dict['kernel'].append(kernel)
        out_dict['train_acc'].append(accuracy_score(y_train, y_pred_train))
        out_dict['train_recall'].append(recall_score(y_train, y_pred_train))
        out_dict['train_f1'].append(f1_score(y_train, y_pred_train))
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

    fig, ax = plt.subplots(3, figsize=(10, 15))
    ax[0].plot(out_df['c'], out_df['train_acc'], label='train_acc')
    ax[0].plot(out_df['c'], out_df['test_acc'], label='test_acc')
    ax[0].set_ylabel('accuracy', fontsize=10)
    ax[0].set_xlabel(f'C', fontsize=10)
    ax[0].legend()
    ax[0].set_title('Accuracy vs C for SVC (kernel = rbf)')

    ax[1].plot(out_df['c'], out_df['train_recall'], label='train_recall')
    ax[1].plot(out_df['c'], out_df['test_recall'], label='test_recall')
    ax[1].set_ylabel('recall', fontsize=10)
    ax[1].set_xlabel('C', fontsize=10)
    ax[1].legend()
    ax[1].set_title('Recall vs C for SVC (kernel = rbf)')

    ax[2].plot(out_df['c'], out_df['train_time'], label='train_time')
    ax[2].set_xlabel('C')
    ax[2].legend()
    ax[2].set_title(f'Train time vs C for SVC (kernel = rbf)')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'svc_scores_vs_C.png'))
    plt.close()


def tune_svc_c():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)
    X_train = X_train[0:int(len(X_train) * 0.2)]
    y_train = y_train[0:int(len(y_train) * 0.2)]

    out_dict = {
        'c': [],
        'train_acc': [],
        'train_recall': [],
        'train_f1': [],
        'test_acc': [],
        'test_recall': [],
        'test_f1': [],
        'train_time': [],
    }

    for c in np.arange(0.1, 1.1, 0.1):
        print(c)
        st = time.time()
        svm_clf = SVC(kernel='linear', random_state=0, C=c, verbose=False)
        svm_clf.fit(X_train, y_train)

        train_time = time.time() - st
        y_pred = svm_clf.predict(X_test)
        y_pred_train = svm_clf.predict(X_train)

        out_dict['c'].append(c)
        out_dict['train_acc'].append(accuracy_score(y_train, y_pred_train))
        out_dict['train_recall'].append(recall_score(y_train, y_pred_train))
        out_dict['train_f1'].append(f1_score(y_train, y_pred_train))
        out_dict['test_acc'].append(accuracy_score(y_test, y_pred))
        out_dict['test_recall'].append(recall_score(y_test, y_pred))
        out_dict['test_f1'].append(f1_score(y_test, y_pred))
        out_dict['train_time'].append(train_time)

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(os.path.join(BASE_DIR, 'results', 'svm_tune_c.csv'), index=False)

    plot_svc_tune_c(out_df)

    return out_df


def svc_best():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)
    X_train = X_train[0:int(len(X_train) * 0.4)]
    y_train = y_train[0:int(len(y_train) * 0.4)]

    st = time.time()
    svm_clf = SVC(kernel='linear', random_state=0, C=0.1, verbose=True)
    svm_clf.fit(X_train, y_train)

    train_time = time.time() - st
    y_pred = svm_clf.predict(X_test)
    y_pred_train = svm_clf.predict(X_train)

    out_dict = {}
    out_dict['train_acc'] = accuracy_score(y_train, y_pred_train)
    out_dict['train_recall'] = recall_score(y_train, y_pred_train)
    out_dict['train_f1'] = f1_score(y_train, y_pred_train)
    out_dict['test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['test_recall'] = recall_score(y_test, y_pred)
    out_dict['test_f1'] = f1_score(y_test, y_pred)
    out_dict['train_time'] = train_time

    return out_dict


def run_svc():
    initial_compare_res = svc_initial_compare()
    print(initial_compare_res)
    tune_svc_kernels()
    tune_svc_c()
    res_best_svc = svc_best()
    print(res_best_svc)
