import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from preprocess_data import preprocess_credit_card_data, normal_train_test_split
from settings import BASE_DIR


def adaBoost_tree_no_prune():
    dt_clf = DecisionTreeClassifier(random_state=0)
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)
    ada_clf_pre_prune = AdaBoostClassifier(dt_clf, random_state=0)
    ada_clf_pre_prune.fit(X_train, y_train)
    y_pred = ada_clf_pre_prune.predict(X_test)

    out_dict = {}

    out_dict['ada_no_prune_test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['ada_no_prune_test_recall'] = recall_score(y_test, y_pred)
    out_dict['ada_no_prune_test_f1'] = f1_score(y_test, y_pred)

    return out_dict


def plot_pre_pruning_adaBoost_tree(param_type='', i: float = 2, j: float = 100, k: float = 1):
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)
    y_train = y_train.reshape(-1,)
    y_test = y_test.reshape(-1,)

    acc_scores = []
    recall_scores = []
    f1_scores = []

    for param_range in np.arange(i, j, k):
        if param_type == 'max_leaf_nodes':
            dt_clf_pre_prune = DecisionTreeClassifier(
                max_leaf_nodes=param_range,
                random_state=0,
                criterion='entropy'
            )
        elif param_type == 'max_depth':
            dt_clf_pre_prune = DecisionTreeClassifier(
                max_depth=param_range,
                random_state=0,
                criterion='entropy'
            )
        elif param_type == 'min_samples_split':
            dt_clf_pre_prune = DecisionTreeClassifier(
                min_samples_split=param_range,
                random_state=0,
                criterion='entropy'
            )
        elif param_type == 'min_samples_leaf':
            dt_clf_pre_prune = DecisionTreeClassifier(
                min_samples_leaf=param_range,
                random_state=0,
                criterion='entropy'
            )
        elif param_type == 'ccp_alpha':
            dt_clf_pre_prune = DecisionTreeClassifier(
                ccp_alpha=param_range,
                random_state=0,
                criterion='entropy'
            )

        if param_type in ['max_leaf_nodes', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'ccp_alpha']:
            ada_clf_pre_prune = AdaBoostClassifier(dt_clf_pre_prune, random_state=0)

        else:
            dt_clf_pre_prune = DecisionTreeClassifier(random_state=0, criterion='entropy')
            if param_type == 'n_estimators':
                ada_clf_pre_prune = AdaBoostClassifier(dt_clf_pre_prune, n_estimators=param_range, random_state=0)
            elif param_type == 'learning_rate':
                ada_clf_pre_prune = AdaBoostClassifier(dt_clf_pre_prune, learning_rate=param_range, random_state=0)

        ada_clf_pre_prune.fit(X_train, y_train)
        y_pred = ada_clf_pre_prune.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    plt.plot(list(np.arange(i, j, k)), acc_scores, label="acc")
    plt.plot(list(np.arange(i, j, k)), recall_scores, label="recall")
    plt.plot(list(np.arange(i, j, k)), f1_scores, label="f1")
    plt.legend()
    plt.title(f'AdaBoost Tree Scores with different {param_type}')
    plt.xlabel(f'{param_type}', fontsize=12)
    plt.ylabel('scores', fontsize=12)
    plt.savefig(os.path.join(BASE_DIR, 'plots', f'AdaBoost_tree_scores_{param_type}'), dpi=300)
    # plt.show()
    plt.close()


def adaBoost_tree_with_prune(run_from_scratch:bool = False):
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)
    y_train = y_train.reshape(-1,)
    y_test = y_test.reshape(-1,)

    dt_clf = DecisionTreeClassifier(
        random_state=0,
        criterion='entropy'
    )

    if run_from_scratch:
        ada_clf_pruned = GridSearchCV(
            estimator= AdaBoostClassifier(dt_clf, random_state=0),
            param_grid=
            {
                'estimator__max_leaf_nodes': np.arange(10, 100, 10),
                'estimator__max_depth': range(2, 20),
                'estimator__min_samples_split': range(10, 30),
                'estimator__min_samples_leaf': range(40, 60),
            },
            scoring='f1',
            verbose=10
        )
        ada_clf_pruned.fit(X_train, y_train)
        print(ada_clf_pruned.best_params_)

        model_name = 'adaBoost_clf_pre_prune_best.pkl'
        with open(os.path.join(BASE_DIR, 'models', model_name), 'wb') as file:
            pickle.dump(ada_clf_pruned, file)

    else:
        model_name = 'adaBoost_clf_pre_prune_best.pkl'
        with open(os.path.join(BASE_DIR, 'models', model_name), 'rb') as file:
            ada_clf_pruned = pickle.load(file)
        print(ada_clf_pruned.best_params_)

    y_pred = ada_clf_pruned.predict(X_test)

    out_dict = {}

    out_dict['best_pre_prune_acc'] = accuracy_score(y_test, y_pred)
    out_dict['best_pre_prune_recall'] = recall_score(y_test, y_pred)
    out_dict['best_pre_prune_f1'] = f1_score(y_test, y_pred)

    return out_dict


def run_adaBoost_trees():
    no_prune_scores = adaBoost_tree_no_prune()

    plot_pre_pruning_adaBoost_tree(param_type='max_leaf_nodes', i=2, j=100)
    plot_pre_pruning_adaBoost_tree(param_type='max_depth', i=2, j=100)
    plot_pre_pruning_adaBoost_tree(param_type='min_samples_split', i=2, j=100)
    plot_pre_pruning_adaBoost_tree(param_type='min_samples_leaf', i=2, j=100)
    plot_pre_pruning_adaBoost_tree(param_type='n_estimators', i=10, j=100, k=10)
    plot_pre_pruning_adaBoost_tree(param_type='learning_rate', i=0.01, j=1.0, k=0.01)

    pruned_scores = adaBoost_tree_with_prune(True)

    print(f'no prune: {no_prune_scores}')
    print(f'pruned_scores: {pruned_scores}')




