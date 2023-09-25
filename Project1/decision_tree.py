import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from preprocess_data import preprocess_credit_card_data, normal_train_test_split
from settings import BASE_DIR


def decision_tree_no_prune():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    # initialize the model with no specification
    # scoring = {
    #     'precision': make_scorer(precision_score, average='weighted'),
    #     'recall': make_scorer(recall_score, average='weighted'),
    #     'f1_score': make_scorer(f1_score, average='weighted')
    # }
    #
    # # Perform 5-fold cross-validation with custom scoring metrics
    # # cv_results = cross_validate(dt_model, X, Y, cv=5)
    # # cv_results = cross_validate(dt_clf, X, Y, cv=5, scoring=scoring)

    dt_clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)

    out_dict = {}

    out_dict['no_prune_acc'] = accuracy_score(y_test, y_pred)
    out_dict['no_prune_recall'] = recall_score(y_test, y_pred)
    out_dict['no_prune_f1'] = f1_score(y_test, y_pred)

    return out_dict


def plot_pre_pruning_decision_tree(param_type='', i: float = 2, j: float = 100, k: float = 1):
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    acc_scores = []
    recall_scores = []
    f1_scores = []
    if param_type == 'ccp_alpha':
        k = 0.01
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
        dt_clf_pre_prune.fit(X_train, y_train)
        y_pred = dt_clf_pre_prune.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    plt.plot(list(np.arange(i, j, k)), acc_scores, label="acc")
    plt.plot(list(np.arange(i, j, k)), recall_scores, label="recall")
    plt.plot(list(np.arange(i, j, k)), f1_scores, label="f1")
    plt.legend()
    plt.title(f'Scores with different {param_type}')
    plt.xlabel(f'{param_type}', fontsize=12)
    plt.ylabel('scores', fontsize=12)
    plt.savefig(os.path.join(BASE_DIR, 'plots', f'decision_tree_scores_{param_type}'), dpi=300)
    # plt.show()
    plt.close()


def decision_tree_with_prune(run_from_scratch:bool = False):
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    if run_from_scratch:
        dt_clf_pruned = GridSearchCV(
            estimator=DecisionTreeClassifier(),
            param_grid=
            {
                'max_leaf_nodes': range(30, 60),
                'max_depth': range(2, 20),
                'min_samples_split': range(10, 30),
                'min_samples_leaf': range(2, 20),
            },
            scoring='f1',
            verbose=10
        )
        dt_clf_pruned.fit(X_train, y_train)
        print(dt_clf_pruned.best_params_)

        model_name = 'dt_clf_pre_prune_best.pkl'
        with open(os.path.join(BASE_DIR, 'models', model_name), 'wb') as file:
            pickle.dump(dt_clf_pruned, file)

    else:
        model_name = 'dt_clf_pre_prune_best.pkl'
        with open(os.path.join(BASE_DIR, 'models', model_name), 'rb') as file:
            dt_clf_pruned = pickle.load(file)

    y_pred = dt_clf_pruned.predict(X_test)

    out_dict = {}

    out_dict['best_pre_prune_acc'] = accuracy_score(y_test, y_pred)
    out_dict['best_pre_prune_recall'] = recall_score(y_test, y_pred)
    out_dict['best_pre_prune_f1'] = f1_score(y_test, y_pred)

    return out_dict


def run_decision_trees():
    no_prune_scores = decision_tree_no_prune()

    plot_pre_pruning_decision_tree(param_type='max_leaf_nodes', i=2, j=100)
    plot_pre_pruning_decision_tree(param_type='max_depth', i=2, j=100)
    plot_pre_pruning_decision_tree(param_type='min_samples_split', i=2, j=100)
    plot_pre_pruning_decision_tree(param_type='min_samples_leaf', i=2, j=100)
    plot_pre_pruning_decision_tree(param_type='ccp_alpha', i=0, j=0.1)

    pruned_scores = decision_tree_with_prune()

    print(f'no prune: {no_prune_scores}')
    print(f'pruned_scores: {pruned_scores}')
