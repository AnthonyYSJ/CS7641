import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from settings import BASE_DIR


def plot_p1_res(res: Dict, problem_name: str):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    algorithms = ['RHC', 'GA', 'SA', 'MIMIC']
    colors = {'RHC': 'b', 'GA': 'r', 'SA': 'c', 'MIMIC': 'y'}
    markers = {'RHC': 'o', 'GA': 's', 'SA': '^', 'MIMIC': 'x'}

    for algo in algorithms:
        ax[0][0].plot(res[algo]['size'], res[algo]['best_fitness'],
                      label=algo, marker=markers[algo], color=colors[algo])
    ax[0][0].set_ylabel('best fitness', fontsize=10)
    ax[0][0].set_xlabel(f'problem size', fontsize=10)
    ax[0][0].legend()
    ax[0][0].set_title(f'{problem_name}: best fitness vs problem size')

    for algo in algorithms:
        ax[0][1].plot(res[algo]['size'], res[algo]['time'],
                      label=algo, marker=markers[algo], color=colors[algo])
    ax[0][1].set_ylabel('time', fontsize=10)
    ax[0][1].set_xlabel(f'problem size', fontsize=10)
    ax[0][1].legend()
    ax[0][1].set_title(f'{problem_name}: time vs problem size')

    for algo in algorithms:
        num_iterations = [len(curve) for curve in res[algo]['fitness_curve']]

        ax[0][2].plot(res[algo]['size'], num_iterations,
                      label=algo, marker=markers[algo], color=colors[algo])
    ax[0][2].set_ylabel('num of iterations', fontsize=10)
    ax[0][2].set_xlabel(f'problem size', fontsize=10)
    ax[0][2].legend()
    ax[0][2].set_title(f'{problem_name}: num of iterations vs problem size')

    for algo in algorithms:
        num_iterations = [len(curve) for curve in res[algo]['fitness_curve']]
        ax[1][0].plot(num_iterations, res[algo]['best_fitness'],
                      label=algo, marker=markers[algo], color=colors[algo])
    ax[1][0].set_ylabel('best fitness', fontsize=10)
    ax[1][0].set_xlabel(f'num of iterations', fontsize=10)
    ax[1][0].legend()
    ax[1][0].set_title(f'{problem_name}: best fitness vs num of iterations')

    for algo in algorithms:
        num_function_evaluations = [curve[-1][-1] for curve in res[algo]['fitness_curve']]
        ax[1][1].plot(res[algo]['size'], num_function_evaluations,
                      label=algo, marker=markers[algo], color=colors[algo])
    ax[1][1].set_ylabel('num of function evaluations', fontsize=10)
    ax[1][1].set_xlabel(f'problem size', fontsize=10)
    ax[1][1].legend()
    ax[1][1].set_title(f'{problem_name}: num of function evaluations vs problem size')

    for algo in algorithms:
        num_function_evaluations = [curve[-1][-1] for curve in res[algo]['fitness_curve']]
        time_per_evaluations = [
            res[algo]['time'][i] / num_function_evaluations[i] for i in range(len(res[algo]['time']))
        ]
        ax[1][2].plot(res[algo]['size'], time_per_evaluations,
                      label=algo, marker=markers[algo], color=colors[algo])
    ax[1][2].set_ylabel('time per evaluation', fontsize=10)
    ax[1][2].set_xlabel(f'problem size', fontsize=10)
    ax[1][2].legend()
    ax[1][2].set_title(f'{problem_name}: time per evaluation vs problem size')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'{problem_name}.png'), bbox_inches='tight')
    plt.close()


def plot_p2_tune_step_size(res: Dict):
    fig, ax = plt.subplots(3, 3, figsize=(20, 16))
    metrics = [['train_accuracy', 'train_recall', 'train_f1'],
               ['test_accuracy', 'test_recall', 'test_f1']]
    colors = {'rhc': 'b', 'ga': 'r', 'sa': 'c'}
    markers = {'rhc': 'o', 'ga': 's', 'sa': 'x'}

    problem_types = ['rhc', 'ga', 'sa']
    for i in range(len(problem_types)):
        problem_type = problem_types[i]
        for step_size, step_res in zip(res[problem_type]['step_size'], res[problem_type]['res']):
            loss_curve = step_res['loss_cur'][:, 0]

            ax[0][i].plot(loss_curve, label=step_size)

        ax[0][i].set_ylabel('loss', fontsize=10)
        ax[0][i].set_xlabel('epoch', fontsize=10)
        ax[0][i].legend()
        ax[0][i].set_title(f'{problem_type}: loss vs epoch')

        for j in range(len(metrics)):
            tmp = metrics[j]
            for k in range(len(tmp)):
                metric = metrics[j][k]
                metric_vals = [le[metric] for le in res[problem_type]['res']]
                ax[j + 1][k].plot(res[problem_type]['step_size'], metric_vals,
                                  label=problem_type, marker=markers[problem_type], color=colors[problem_type])

    for i in range(len(metrics)):
        tmp = metrics[i]
        for j in range(len(tmp)):
            metric = metrics[i][j]
            ax[i + 1][j].set_ylabel(metric, fontsize=10)
            ax[i + 1][j].set_xlabel('step_size', fontsize=10)
            ax[i + 1][j].legend()
            ax[i + 1][j].set_title(f'{metric} vs step_size')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'nn_tune_step_size.png'), bbox_inches='tight')
    plt.close()


def plot_p2_tune_step_size_time(res: Dict):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    colors = {'rhc': 'b', 'ga': 'r', 'sa': 'c'}
    markers = {'rhc': 'o', 'ga': 's', 'sa': 'x'}

    problem_types = ['rhc', 'ga', 'sa']

    for problem_type in problem_types:
        num_epochs = [len(x['loss_cur'][:, 0]) for x in res[problem_type]['res']]
        fit_times = [x['fit_time'] for x in res[problem_type]['res']]
        fit_time_per_epoch = [fit_times[i]/num_epochs[i] for i in range(len(num_epochs))]
        ax[0].plot(res[problem_type]['step_size'], fit_times,
                   label=problem_type, marker=markers[problem_type], color=colors[problem_type])
        ax[1].plot(res[problem_type]['step_size'], fit_time_per_epoch,
                   label=problem_type, marker=markers[problem_type], color=colors[problem_type])
        ax[2].plot(res[problem_type]['step_size'], num_epochs,
                   label=problem_type, marker=markers[problem_type], color=colors[problem_type])

    ax[0].set_ylabel('fit time', fontsize=10)
    ax[0].set_xlabel('step_size', fontsize=10)
    ax[0].legend()
    ax[0].set_title(f'time vs step_size')

    ax[1].set_ylabel('fit time per epoch', fontsize=10)
    ax[1].set_xlabel('step_size', fontsize=10)
    ax[1].legend()
    ax[1].set_title(f'time per epoch vs step_size')

    ax[2].set_ylabel('num of epochs', fontsize=10)
    ax[2].set_xlabel('step_size', fontsize=10)
    ax[2].legend()
    ax[2].set_title(f'num of epoch vs step_size')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'nn_tune_step_size_time.png'), bbox_inches='tight')
    plt.close()


def plot_p2_tune_gd_lr(res: Dict):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    for lr, step_res in zip(res['lr'], res['res']):
        loss_curve = step_res['loss_cur']
        ax[0][0].plot(loss_curve, label=lr)

    ax[0][0].set_ylabel('loss', fontsize=10)
    ax[0][0].set_xlabel('epoch', fontsize=10)
    ax[0][0].legend()
    ax[0][0].set_title(f'gd: loss vs epoch')

    log_lr = [np.log10(x) for x in res['lr']]
    train_accuracy = [x['train_accuracy'] for x in res['res']]
    test_accuracy = [x['test_accuracy'] for x in res['res']]
    train_recall = [x['train_recall'] for x in res['res']]
    test_recall = [x['test_recall'] for x in res['res']]
    train_f1 = [x['train_f1'] for x in res['res']]
    test_f1 = [x['test_f1'] for x in res['res']]
    fit_time = [x['fit_time'] for x in res['res']]
    num_epochs = [len(x['loss_cur']) for x in res['res']]
    fit_time_per_epoch = [fit_time[i]/num_epochs[i] for i in range(len(num_epochs))]

    ax[0][1].plot(log_lr, train_accuracy, label='train_accuracy')
    ax[0][1].plot(log_lr, test_accuracy, label='test_accuracy')
    ax[0][1].set_ylabel('accuracy', fontsize=10)
    ax[0][1].set_xlabel('log10(lr)', fontsize=10)
    ax[0][1].legend()
    ax[0][1].set_title(f'gd: accuracy vs log10(lr)')

    ax[0][2].plot(log_lr, train_recall, label='train_recall')
    ax[0][2].plot(log_lr, test_recall, label='test_recall')
    ax[0][2].set_ylabel('recall', fontsize=10)
    ax[0][2].set_xlabel('log10(lr)', fontsize=10)
    ax[0][2].legend()
    ax[0][2].set_title(f'gd: recall vs log10(lr)')

    ax[1][0].plot(log_lr, train_f1, label='train_f1')
    ax[1][0].plot(log_lr, test_f1, label='test_f1')
    ax[1][0].set_ylabel('f1 score', fontsize=10)
    ax[1][0].set_xlabel('log10(lr)', fontsize=10)
    ax[1][0].legend()
    ax[1][0].set_title(f'gd: f1 score vs log10(lr)')

    ax[1][1].plot(log_lr, fit_time, label='fit_time')
    ax[1][1].set_ylabel('fit_time', fontsize=10)
    ax[1][1].set_xlabel('log10(lr)', fontsize=10)
    ax[1][1].legend()
    ax[1][1].set_title(f'gd: fit_time vs log10(lr)')

    ax[1][2].plot(log_lr, fit_time_per_epoch, label='fit_time')
    ax[1][2].set_ylabel('fit_time_per_epoch', fontsize=10)
    ax[1][2].set_xlabel('log10(lr)', fontsize=10)
    ax[1][2].legend()
    ax[1][2].set_title(f'gd: fit_time_per_epoch vs log10(lr)')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'nn_tune_gd_lr.png'), bbox_inches='tight')
    plt.close()
