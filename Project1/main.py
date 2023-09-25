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
from knn import run_knn

from preprocess_data import preprocess_credit_card_data, normal_train_test_split


if __name__ == "__main__":
    run_decision_trees()
    run_adaBoost_trees()
    run_neural_network()
    run_svc()
    run_knn()


    print("Done!")
