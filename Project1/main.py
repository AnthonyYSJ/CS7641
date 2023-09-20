import os
import sklearn
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time

from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from settings import BASE_DIR

from decision_tree import run_decision_trees
from adaBoost_tree import run_adaBoost_trees
from neural_network import run_neural_network
from svc import run_svc
from preprocess_data import preprocess_credit_card_data, normal_train_test_split


def get_knn_res_comparing_norm():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    knn_clf = KNeighborsClassifier()

    st = time.time()
    knn_clf.fit(X_train, y_train)

    train_time = time.time() - st

    y_pred = knn_clf.predict(X_test)

    out_dict = {'wo_norm': {}, 'norm': {}}
    out_dict['wo_norm']['train_acc'] = knn_clf.score(X_train, y_train)
    out_dict['wo_norm']['test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['wo_norm']['test_recall'] = recall_score(y_test, y_pred)
    out_dict['wo_norm']['test_f1'] = f1_score(y_test, y_pred)
    out_dict['wo_norm']['train_time'] = train_time

    # then with normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    knn_clf = KNeighborsClassifier()

    st = time.time()
    knn_clf.fit(X_train, y_train)

    train_time = time.time() - st

    y_pred = knn_clf.predict(X_test)

    out_dict['norm']['train_acc'] = knn_clf.score(X_train, y_train)
    out_dict['norm']['test_acc'] = accuracy_score(y_test, y_pred)
    out_dict['norm']['test_recall'] = recall_score(y_test, y_pred)
    out_dict['norm']['test_f1'] = f1_score(y_test, y_pred)
    out_dict['norm']['train_time'] = train_time

    return out_dict


def run_knn():
    knn_res_comparing_norm = get_knn_res_comparing_norm()

if __name__ == "__main__":
    # run_decision_trees()
    # run_adaBoost_trees()
    # run_neural_network()
    # run_svc()




    print("Done!")
