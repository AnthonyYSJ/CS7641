import mlrose_hiive as mlrose
import os
import pickle
from preprocess_data import preprocess_credit_card_data, normal_train_test_split
from p2_utils import run_p2_algo, print_best_gd_res, print_best_ro_res
from plotter import plot_p2_tune_step_size, plot_p2_tune_step_size_time, plot_p2_tune_gd_lr
from settings import BASE_DIR


def run_gd(lr: float):
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    problem_type = 'gd'
    num_epoch = 1000

    gd_clf = mlrose.NeuralNetwork(hidden_nodes=[512, 128, 16], activation='relu',
                                  algorithm='gradient_descent', max_iters=num_epoch, bias=True, is_classifier=True,
                                  learning_rate=lr, early_stopping=False, clip_max=1e+10, curve=True)
    result = run_p2_algo(gd_clf, X_train, X_test, y_train, y_test, problem_type, num_epoch, lr)

    return result


def run_rhc(step_size: int = 5, num_epoch: int = 1000, max_attempts: int = 10, restarts: int = 10):
    problem_type = 'rhc'

    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    rhc_clf = mlrose.NeuralNetwork(hidden_nodes=[512, 128, 16], activation='relu',
                                   algorithm='random_hill_climb', max_iters=num_epoch, bias=True, is_classifier=True,
                                   learning_rate=step_size, early_stopping=True, clip_max=1e+10, restarts=restarts,
                                   max_attempts=max_attempts, curve=True)
    result = run_p2_algo(rhc_clf, X_train, X_test, y_train, y_test, problem_type, num_epoch, step_size)

    return result


def run_ga(step_size: int = 5, num_epoch: int = 10, max_attempts: int = 10):
    problem_type = 'ga'

    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    ga_clf = mlrose.NeuralNetwork(hidden_nodes=[512, 128, 16], activation='relu',
                                  algorithm='genetic_alg', max_iters=num_epoch, bias=True, is_classifier=True,
                                  learning_rate=step_size, early_stopping=True, clip_max=1e+10, mutation_prob=0.4,
                                  max_attempts=max_attempts, curve=True)

    result = run_p2_algo(ga_clf, X_train, X_test, y_train, y_test, problem_type, num_epoch, step_size)

    return result


def run_sa(step_size: int = 5, num_epoch: int = 1000, max_attempts: int = 10):
    problem_type = 'sa'

    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y)

    sa_clf = mlrose.NeuralNetwork(hidden_nodes=[512, 128, 16], activation='relu',
                                  algorithm='simulated_annealing', max_iters=num_epoch, bias=True, is_classifier=True,
                                  learning_rate=step_size, early_stopping=True, clip_max=1e+10,
                                  schedule=mlrose.GeomDecay(),
                                  max_attempts=max_attempts, curve=True)

    result = run_p2_algo(sa_clf, X_train, X_test, y_train, y_test, problem_type, num_epoch, step_size)

    return result


def tune_step_size():
    results = {
        'rhc': {'step_size': [], 'res': []},
        'ga': {'step_size': [], 'res': []},
        'sa': {'step_size': [], 'res': []}
    }

    for problem_type in ['rhc', 'ga', 'sa']:
        for step_size in range(5, 30, 5):
            print(f'problem_type = {problem_type}, step_size = {step_size}')
            results[problem_type]['step_size'].append(step_size)
            if problem_type == 'rhc':
                results[problem_type]['res'].append(run_rhc(step_size))
            elif problem_type == 'ga':
                results[problem_type]['res'].append(run_ga(step_size))
            else:  # problem_type == 'sa'
                results[problem_type]['res'].append(run_sa(step_size))

    with open(os.path.join(BASE_DIR, 'results', f'nn_tune_step_size.pkl'), 'wb') as my_file:
        pickle.dump(results, my_file)

    return results


def tune_lr():
    results = {'lr': [], 'res': []}

    for lr in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        print(f'problem_type = gd, lr = {lr}')
        results['lr'].append(lr)
        results['res'].append(run_gd(lr))

    with open(os.path.join(BASE_DIR, 'results', f'nn_tune_lr.pkl'), 'wb') as my_file:
        pickle.dump(results, my_file)

    return results


def run_nn():
    ro_res = tune_step_size()
    gd_res = tune_lr()

    plot_p2_tune_step_size(ro_res)
    plot_p2_tune_step_size_time(ro_res)
    plot_p2_tune_gd_lr(gd_res)

    print_best_ro_res()
    print_best_gd_res()
