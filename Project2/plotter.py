import os
import matplotlib.pyplot as plt

from typing import Dict
from settings import BASE_DIR


def plot_res(res: Dict, problem_name: str):
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
            res[algo]['time'][i]/num_function_evaluations[i] for i in range(len(res[algo]['time']))
        ]
        ax[1][2].plot(res[algo]['size'], time_per_evaluations,
                      label=algo, marker=markers[algo], color=colors[algo])
    ax[1][2].set_ylabel('time per evaluation', fontsize=10)
    ax[1][2].set_xlabel(f'problem size', fontsize=10)
    ax[1][2].legend()
    ax[1][2].set_title(f'{problem_name}: time per evaluation vs problem size')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'{problem_name}.png'), bbox_inches='tight')
    plt.close()
