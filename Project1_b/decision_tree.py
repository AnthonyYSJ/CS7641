import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from preprocess_data import preprocess_credit_card_data, normal_train_test_split, smote_train_test_split
from settings import BASE_DIR


def train_decision_tree(model, X_train, X_test, y_train, y_test):
    st = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - st

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    out_dict = {}

    out_dict['train_acc'] = accuracy_score(y_train, y_pred_train)
    out_dict['train_recall'] = recall_score(y_train, y_pred_train)
    out_dict['train_f1'] = f1_score(y_train, y_pred_train)
    out_dict['test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['test_recall'] = recall_score(y_test, y_pred)
    out_dict['test_f1'] = f1_score(y_test, y_pred)
    out_dict['train_time'] = train_time
    return out_dict


def decision_tree_no_prune_imbalanced():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    dt_clf = DecisionTreeClassifier(random_state=0, criterion='entropy')

    out_dict = train_decision_tree(dt_clf, X_train, X_test, y_train, y_test)

    return out_dict


def decision_tree_no_prune_balanced():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)

    dt_clf = DecisionTreeClassifier(random_state=0, criterion='entropy')

    out_dict = train_decision_tree(dt_clf, X_train, X_test, y_train, y_test)

    return out_dict


def plot_decision_tree_res(out_dict):
    assert len(set(out_dict['param_type'])) == 1
    param_type = list(set(out_dict['param_type']))[0]
    fig, ax = plt.subplots(3, figsize=(10, 15))
    ax[0].plot(out_dict['param_val'], out_dict['train_acc'], label='train_acc')
    ax[0].plot(out_dict['param_val'], out_dict['test_acc'], label='test_acc')
    ax[0].set_ylabel('accuracy', fontsize=10)
    ax[0].set_xlabel(f'{param_type}', fontsize=10)
    ax[0].legend()
    ax[0].set_title(f'decision tree accuracy vs {param_type}')

    ax[1].plot(out_dict['param_val'], out_dict['train_recall'], label='train_recall')
    ax[1].plot(out_dict['param_val'], out_dict['test_recall'], label='test_recall')
    ax[1].set_ylabel('recall', fontsize=10)
    ax[1].set_xlabel(f'{param_type}', fontsize=10)
    ax[1].legend()
    ax[1].set_title(f'decision tree recall vs {param_type}')

    ax[2].plot(out_dict['param_val'], out_dict['train_time'], label='train_time')
    ax[2].set_xlabel(f'{param_type}', fontsize=10)
    ax[2].legend()
    ax[2].set_title(f'decision tree train_time vs {param_type}')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'decision_tree_scores_{param_type}.png'))
    plt.close()


def prune_decision_tree(param_type='', i: float = 2, j: float = 100, k: float = 1):
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
    
    if param_type == 'ccp_alpha':
        k = 0.01
    for param_val in np.arange(i, j, k):
        print(f'{param_type} = {param_val}')

        if param_type == 'max_leaf_nodes':
            dt_clf = DecisionTreeClassifier(
                max_leaf_nodes=param_val,
                random_state=0,
                criterion='entropy'
            )
        elif param_type == 'max_depth':
            dt_clf = DecisionTreeClassifier(
                max_depth=param_val,
                random_state=0,
                criterion='entropy'
            )
        elif param_type == 'min_samples_split':
            dt_clf = DecisionTreeClassifier(
                min_samples_split=param_val,
                random_state=0,
                criterion='entropy'
            )
        elif param_type == 'min_samples_leaf':
            dt_clf = DecisionTreeClassifier(
                min_samples_leaf=param_val,
                random_state=0,
                criterion='entropy'
            )
        else:  # param_type == 'ccp_alpha':
            dt_clf = DecisionTreeClassifier(
                ccp_alpha=param_val,
                random_state=0,
                criterion='entropy'
            )

        res = train_decision_tree(dt_clf, X_train, X_test, y_train, y_test)
        out_dict['param_type'].append(param_type)
        out_dict['param_val'].append(param_val)
        out_dict['train_acc'].append(res['train_acc'])
        out_dict['train_recall'].append(res['train_recall'])
        out_dict['train_f1'].append(res['train_f1'])
        out_dict['test_acc'].append(res['test_acc'])
        out_dict['test_recall'].append(res['test_recall'])
        out_dict['test_f1'].append(res['test_f1'])
        out_dict['train_time'].append(res['train_time'])

    plot_decision_tree_res(out_dict)
    df = pd.DataFrame(out_dict)
    df.to_csv(os.path.join(BASE_DIR, 'results', f'decision_tree_tune_{param_type}.csv'), index=False)


def decision_tree_prune_best(max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, ccp_alpha):
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y)

    dt_clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
        max_leaf_nodes=max_leaf_nodes,
        ccp_alpha=ccp_alpha
    )

    out_dict = train_decision_tree(dt_clf, X_train, X_test, y_train, y_test)

    return out_dict


def run_decision_trees():
    res_no_prune_im = decision_tree_no_prune_imbalanced()
    print(res_no_prune_im)
    res_no_prune = decision_tree_no_prune_balanced()
    print(res_no_prune)

    prune_decision_tree(param_type='max_leaf_nodes', i=2, j=100)
    prune_decision_tree(param_type='max_depth', i=2, j=100)
    prune_decision_tree(param_type='min_samples_split', i=2, j=100)
    prune_decision_tree(param_type='min_samples_leaf', i=2, j=100)
    prune_decision_tree(param_type='ccp_alpha', i=0, j=0.1)

    pruned_res = decision_tree_prune_best(
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=20,
        max_leaf_nodes=10,
        ccp_alpha=0.01
    )

    print(pruned_res)
