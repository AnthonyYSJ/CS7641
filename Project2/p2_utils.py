import os
import time
import pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score
from settings import BASE_DIR
from typing import Dict


def run_p2_algo(clf, X_train, X_test, y_train, y_test, problem_type, num_epoch, lr):
    st = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - st

    result = {}
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    result['test_accuracy'] = accuracy_score(y_test, y_test_pred)
    result['test_recall'] = recall_score(y_test, y_test_pred)
    result['test_f1'] = f1_score(y_test, y_test_pred)
    result['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    result['train_recall'] = recall_score(y_train, y_train_pred)
    result['train_f1'] = f1_score(y_train, y_train_pred)

    result['fit_time'] = fit_time
    result['num_epoch'] = num_epoch
    result['lr'] = lr
    result['loss_cur'] = clf.fitness_curve

    with open(os.path.join(BASE_DIR, 'results', f'nn_{problem_type}_run_results.pkl'), 'wb') as my_file:
        pickle.dump(result, my_file)

    return result


def get_best_with_algo_res(result: Dict, step_lr_type: str = 'lr'):
    best_f1 = -float("inf")
    best_lr = 0
    best_index = -1
    for i in range(len(result[step_lr_type])):
        if result['res'][i]['test_f1'] > best_f1:
            best_f1 = result['res'][i]['test_f1']
            best_lr = result[step_lr_type][i]
            best_index = i

    accuracy = result['res'][best_index]['test_accuracy']
    recall = result['res'][best_index]['test_recall']
    fit_time = result['res'][best_index]['fit_time']

    return accuracy, recall, best_f1, best_lr, fit_time


def print_best_gd_res():
    with open(os.path.join(BASE_DIR, 'results', f'nn_tune_lr.pkl'), 'rb') as file:
        result = pickle.load(file)
    accuracy, recall, f1, lr, fit_time = get_best_with_algo_res(result, 'lr')

    print(f'algo = gd, best_accuracy = {accuracy}, lr = {lr}, fit_time = {fit_time}')
    print(f'algo = gd, best_recall = {recall}, lr = {lr}, fit_time = {fit_time}')
    print(f'algo = gd, best_f1 = {f1}, lr = {lr}, fit_time = {fit_time}')
    print(f'algo = gd, fit_time = {fit_time}, lr = {lr}, fit_time = {fit_time}')


def print_best_ro_res():
    with open(os.path.join(BASE_DIR, 'results', f'nn_tune_step_size.pkl'), 'rb') as file:
        result = pickle.load(file)

    for algo in result:
        accuracy, recall, f1, lr, fit_time = get_best_with_algo_res(result[algo], 'step_size')
        print(f'algo = {algo}, best_accuracy = {accuracy}, lr = {lr}, fit_time = {fit_time}')
        print(f'algo = {algo}, best_recall = {recall}, lr = {lr}, fit_time = {fit_time}')
        print(f'algo = {algo}, best_f1 = {f1}, lr = {lr}, fit_time = {fit_time}')
        print(f'algo = {algo}, fit_time = {fit_time}, lr = {lr}, fit_time = {fit_time}')
