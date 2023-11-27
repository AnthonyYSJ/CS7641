import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.helper import get_optimal_policy

from settings import BASE_DIR


def plot_vi_pi_convergence(run_name: str, var_type: str, ):
    try:
        assert run_name in ['frozen_lake_vi', 'frozen_lake_pi', 'black_jack_vi', 'black_jack_pi']
    except ValueError:
        print("run_name has to be one of ['frozen_lake_vi', 'frozen_lake_pi', 'black_jack_vi', 'black_jack_pi']!")
        raise

    try:
        assert var_type in ['gamma', 'theta']
    except ValueError:
        print("var_type has to be either gamma or theta!")
        raise

    with open(os.path.join(BASE_DIR, 'results', f'{run_name}.pkl'), 'rb') as file:
        results = pickle.load(file)
    if var_type == 'gamma':
        fixed_var = 'gamma'
        varied_var = 'theta'
        if run_name == 'frozen_lake_vi':
            fixed_var_val = 0.90
        elif run_name == 'frozen_lake_pi':
            fixed_var_val = 0.90
        elif run_name == 'black_jack_vi':
            fixed_var_val = 0.90
        else:  # run_name == 'black_jack_pi':
            fixed_var_val = 0.80
        varied_var_vals = [10**(-i) for i in range(2, 11, 1)]
        varied_var_vals_plot = [np.log10(i) for i in varied_var_vals]
        varied_var_name_plot = f'log10({varied_var})'
    else:  # var_type == 'theta'
        fixed_var = 'theta'
        varied_var = 'gamma'
        if run_name == 'frozen_lake_vi':
            fixed_var_val = 10**(-7)
        elif run_name == 'frozen_lake_pi':
            fixed_var_val = 10**(-5)
        elif run_name == 'black_jack_vi':
            fixed_var_val = 10**(-8)
        else:  # run_name == 'black_jack_pi':
            fixed_var_val = 10**(-7)
        varied_var_vals = [np.round(i, 2) for i in np.arange(0.1, 1.0, 0.1)]
        varied_var_vals_plot = varied_var_vals
        varied_var_name_plot = varied_var

    if run_name.startswith('black_jack'):
        V_plot_range = 20
    else:
        V_plot_range = 200
    mean_scores = []
    var_scores = []
    num_iter_to_converge_list = []
    fit_times = []
    fig, ax = plt.subplots(1, 4, figsize=(25, 5))
    for varied_var_val in varied_var_vals:
        if fixed_var == 'theta':
            spec = f'gamma={varied_var_val}_theta={fixed_var_val}_max_iter=500'
        else:
            spec = f'gamma={fixed_var_val}_theta={varied_var_val}_max_iter=500'
        spec_res = results[spec]
        num_iter_to_converge = spec_res['num_iter_to_converge']
        V_track = spec_res['V_track']
        V_max = np.max(V_track, axis=1)
        V_max[num_iter_to_converge-1:] = V_max[num_iter_to_converge-2]
        num_iter_to_converge_list.append(num_iter_to_converge)
        fit_times.append(spec_res['fit_time'])
        mean_scores.append(np.mean(spec_res['scores']))
        var_scores.append(np.var(spec_res['scores']))
        ax[0].plot(V_max[:V_plot_range], label=f'{varied_var} = {varied_var_val}')
    ax[0].set_ylabel('max of V', fontsize=10)
    ax[0].set_xlabel(f'num_iterations', fontsize=10)
    ax[0].set_title('max of V vs num_iterations', fontsize=10)
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(varied_var_vals_plot, num_iter_to_converge_list)
    ax[1].set_ylabel('number of iterations', fontsize=10)
    ax[1].set_xlabel(f'{varied_var_name_plot}', fontsize=10)
    ax[1].set_title(f'iterations to convergence vs {varied_var_name_plot}, {fixed_var}={fixed_var_val}', fontsize=10)
    ax[1].grid()

    ax[2].plot(varied_var_vals_plot, fit_times)
    ax[2].set_ylabel('fit_time(s)', fontsize=10)
    ax[2].set_xlabel(f'{varied_var_name_plot}', fontsize=10)
    ax[2].set_title(f'fit_time vs {varied_var_name_plot}, {fixed_var}={fixed_var_val}', fontsize=10)
    ax[2].grid()

    ax_dup = ax[3].twinx()
    ax[3].plot(varied_var_vals_plot, mean_scores, label='mean_scores', color='r', marker='o',)
    ax[3].set_ylabel('scores (mean)', fontsize=10, color='r')
    ax_dup.plot(varied_var_vals_plot, var_scores, label='var_scores', color='b', marker='x')
    ax_dup.set_ylabel('scores (variance)', fontsize=10, color='b')
    ax[3].set_xlabel(f'{varied_var_name_plot}', fontsize=10)
    ax[3].set_title(f'scores (mean & variance)  vs {varied_var_name_plot}, {fixed_var}={fixed_var_val}', fontsize=10)
    ax[3].grid()

    fig.savefig(
        os.path.join(BASE_DIR, 'plots', f"{run_name}_{varied_var}_plot.png"),
        bbox_inches='tight',
        transparent=False
    )


def plot_black_jack_ql_convergence():
    with open(os.path.join(BASE_DIR, 'results', 'black_jack_q_learning.pkl'), 'rb') as file:
        results = pickle.load(file)

    gamma_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_list = [0.01, 0.05, 0.10]
    epsilon_list = [0.01, 0.05, 0.10]
    Q_plot_range = 100

    gamma_list_plot = [str(i) for i in gamma_list]
    alpha_list_plot = [str(i) for i in alpha_list]
    epsilon_list_plot = [str(i) for i in epsilon_list]

    fig, ax = plt.subplots(3, 3, figsize=(18, 14))

    fit_times = []
    mean_scores = []
    var_scores = []
    for min_epsilon in epsilon_list:
        spec = f'gamma_0.9_min_alpha_0.05_min_epsilon_{min_epsilon}'
        spec_res = results[spec]
        Q_track = spec_res['Q_track']
        Q_max = np.max(Q_track, axis=2)
        Q_max = np.max(Q_max, axis=1)
        fit_times.append(spec_res['fit_time'])
        mean_scores.append(np.mean(spec_res['scores']))
        var_scores.append(np.var(spec_res['scores']))
        ax[0][0].plot(Q_max[:Q_plot_range], label=f'min_epsilon = {min_epsilon}')

    ax[0][0].set_ylabel('max of Q', fontsize=10)
    ax[0][0].set_xlabel(f'num_iterations', fontsize=10)
    ax[0][0].set_title('max of Q vs n_episodes', fontsize=10)
    ax[0][0].legend()
    ax[0][0].grid()

    ax[0][1].plot(epsilon_list_plot, fit_times)
    ax[0][1].set_ylabel('fit_time(s)', fontsize=10)
    ax[0][1].set_xlabel('min_epsilon', fontsize=10)
    ax[0][1].set_title('fit_time vs min_epsilon, gamma=0.9, min_alpha=0.05', fontsize=10)
    ax[0][1].grid()

    ax_dup = ax[0][2].twinx()
    ax[0][2].plot(epsilon_list_plot, mean_scores, label='mean_scores', color='r', marker='o',)
    ax[0][2].set_ylabel('scores (mean)', fontsize=10, color='r')
    ax_dup.plot(epsilon_list_plot, var_scores, label='var_scores', color='b', marker='x')
    ax_dup.set_ylabel('scores (variance)', fontsize=10, color='b')
    ax[0][2].set_xlabel('min_epsilon', fontsize=10)
    ax[0][2].set_title('scores (mean & variance) vs min_epsilon, gamma=0.9, min_alpha=0.05', fontsize=10)
    ax[0][2].grid()

    fit_times = []
    mean_scores = []
    var_scores = []
    for gamma in gamma_list:
        spec = f'gamma_{gamma}_min_alpha_0.05_min_epsilon_0.1'
        spec_res = results[spec]
        Q_track = spec_res['Q_track']
        Q_max = np.max(Q_track, axis=2)
        Q_max = np.max(Q_max, axis=1)
        fit_times.append(spec_res['fit_time'])
        mean_scores.append(np.mean(spec_res['scores']))
        var_scores.append(np.var(spec_res['scores']))
        ax[1][0].plot(Q_max[:Q_plot_range], label=f'gamma = {gamma}')

    ax[1][0].set_ylabel('max of Q', fontsize=10)
    ax[1][0].set_xlabel(f'num_iterations', fontsize=10)
    ax[1][0].set_title('max of Q vs n_episodes', fontsize=10)
    ax[1][0].legend()
    ax[1][0].grid()

    ax[1][1].plot(gamma_list_plot, fit_times)
    ax[1][1].set_ylabel('fit_time(s)', fontsize=10)
    ax[1][1].set_xlabel('gamma', fontsize=10)
    ax[1][1].set_title('fit_time vs gamma, min_alpha=0.05, min_epsilon=0.1', fontsize=10)
    ax[1][1].grid()

    ax_dup = ax[1][2].twinx()
    ax[1][2].plot(gamma_list_plot, mean_scores, label='mean_scores', color='r', marker='o',)
    ax[1][2].set_ylabel('scores (mean)', fontsize=10, color='r')
    ax_dup.plot(gamma_list_plot, var_scores, label='var_scores', color='b', marker='x')
    ax_dup.set_ylabel('scores (variance)', fontsize=10, color='b')
    ax[1][2].set_xlabel('gamma', fontsize=10)
    ax[1][2].set_title('scores (mean & variance) vs gamma, min_alpha=0.05, min_epsilon=0.1', fontsize=10)
    ax[1][2].grid()

    fit_times = []
    mean_scores = []
    var_scores = []
    for min_alpha in alpha_list:
        spec = f'gamma_0.1_min_alpha_{min_alpha}_min_epsilon_0.1'
        spec_res = results[spec]
        Q_track = spec_res['Q_track']
        Q_max = np.max(Q_track, axis=2)
        Q_max = np.max(Q_max, axis=1)
        fit_times.append(spec_res['fit_time'])
        mean_scores.append(np.mean(spec_res['scores']))
        var_scores.append(np.var(spec_res['scores']))
        ax[2][0].plot(Q_max[:Q_plot_range], label=f'min_alpha = {min_alpha}')

    ax[2][0].set_ylabel('max of Q', fontsize=10)
    ax[2][0].set_xlabel(f'num_iterations', fontsize=10)
    ax[2][0].set_title('max of Q vs n_episodes', fontsize=10)
    ax[2][0].legend()
    ax[2][0].grid()

    ax[2][1].plot(alpha_list_plot, fit_times)
    ax[2][1].set_ylabel('fit_time(s)', fontsize=10)
    ax[2][1].set_xlabel('min_alpha', fontsize=10)
    ax[2][1].set_title('fit_time vs min_alpha, gamma=0.1, min_epsilon=0.1', fontsize=10)
    ax[2][1].grid()

    ax_dup = ax[2][2].twinx()
    ax[2][2].plot(alpha_list_plot, mean_scores, label='mean_scores', color='r', marker='o',)
    ax[2][2].set_ylabel('scores (mean)', fontsize=10, color='r')
    ax_dup.plot(alpha_list_plot, var_scores, label='var_scores', color='b', marker='x')
    ax_dup.set_ylabel('scores (variance)', fontsize=10, color='b')
    ax[2][2].set_xlabel('min_alpha', fontsize=10)
    ax[2][2].set_title('scores (mean & variance) vs min_alpha, gamma=0.1, min_epsilon=0.1', fontsize=10)
    ax[2][2].grid()

    fig.savefig(
        os.path.join(BASE_DIR, 'plots', "black_jack_q_learning_plot.png"),
        bbox_inches='tight',
        transparent=False
    )


def plot_frozen_lake_ql_convergence():
    with open(os.path.join(BASE_DIR, 'results', 'frozen_lake_q_learning.pkl'), 'rb') as file:
        results = pickle.load(file)

    gamma_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_list = [0.01, 0.05, 0.10]
    epsilon_list = [0.01, 0.05, 0.10]
    Q_plot_range = 50000

    gamma_list_plot = [str(i) for i in gamma_list]
    alpha_list_plot = [str(i) for i in alpha_list]
    epsilon_list_plot = [str(i) for i in epsilon_list]

    fig, ax = plt.subplots(3, 3, figsize=(18, 14))

    fit_times = []
    mean_scores = []
    var_scores = []
    for min_epsilon in epsilon_list:
        spec = f'gamma_0.9_min_alpha_0.05_min_epsilon_{min_epsilon}'
        spec_res = results[spec]
        Q_track = spec_res['Q_track']
        Q_max = np.max(Q_track, axis=2)
        Q_max = np.max(Q_max, axis=1)
        fit_times.append(spec_res['fit_time'])
        mean_scores.append(np.mean(spec_res['scores']))
        var_scores.append(np.var(spec_res['scores']))
        ax[0][0].plot(Q_max[:Q_plot_range], label=f'min_epsilon = {min_epsilon}')

    ax[0][0].set_ylabel('max of Q', fontsize=10)
    ax[0][0].set_xlabel(f'num_iterations', fontsize=10)
    ax[0][0].set_title('max of Q vs n_episodes', fontsize=10)
    ax[0][0].legend()
    ax[0][0].grid()

    ax[0][1].plot(epsilon_list_plot, fit_times)
    ax[0][1].set_ylabel('fit_time(s)', fontsize=10)
    ax[0][1].set_xlabel('min_epsilon', fontsize=10)
    ax[0][1].set_title('fit_time vs min_epsilon, gamma=0.9, min_alpha=0.05', fontsize=10)
    ax[0][1].grid()

    ax_dup = ax[0][2].twinx()
    ax[0][2].plot(epsilon_list_plot, mean_scores, label='mean_scores', color='r', marker='o',)
    ax[0][2].set_ylabel('scores (mean)', fontsize=10, color='r')
    ax_dup.plot(epsilon_list_plot, var_scores, label='var_scores', color='b', marker='x')
    ax_dup.set_ylabel('scores (variance)', fontsize=10, color='b')
    ax[0][2].set_xlabel('min_epsilon', fontsize=10)
    ax[0][2].set_title('scores (mean & variance) vs min_epsilon, gamma=0.9, min_alpha=0.05', fontsize=10)
    ax[0][2].grid()

    fit_times = []
    mean_scores = []
    var_scores = []
    for gamma in gamma_list:
        spec = f'gamma_{gamma}_min_alpha_0.05_min_epsilon_0.05'
        spec_res = results[spec]
        Q_track = spec_res['Q_track']
        Q_max = np.max(Q_track, axis=2)
        Q_max = np.max(Q_max, axis=1)
        fit_times.append(spec_res['fit_time'])
        mean_scores.append(np.mean(spec_res['scores']))
        var_scores.append(np.var(spec_res['scores']))
        ax[1][0].plot(Q_max[:Q_plot_range], label=f'gamma = {gamma}')

    ax[1][0].set_ylabel('max of Q', fontsize=10)
    ax[1][0].set_xlabel(f'num_iterations', fontsize=10)
    ax[1][0].set_title('max of Q vs n_episodes', fontsize=10)
    ax[1][0].legend()
    ax[1][0].grid()

    ax[1][1].plot(gamma_list_plot, fit_times)
    ax[1][1].set_ylabel('fit_time(s)', fontsize=10)
    ax[1][1].set_xlabel('gamma', fontsize=10)
    ax[1][1].set_title('fit_time vs gamma, min_alpha=0.05, min_epsilon=0.05', fontsize=10)
    ax[1][1].grid()

    ax_dup = ax[1][2].twinx()
    ax[1][2].plot(gamma_list_plot, mean_scores, label='mean_scores', color='r', marker='o',)
    ax[1][2].set_ylabel('scores (mean)', fontsize=10, color='r')
    ax_dup.plot(gamma_list_plot, var_scores, label='var_scores', color='b', marker='x')
    ax_dup.set_ylabel('scores (variance)', fontsize=10, color='b')
    ax[1][2].set_xlabel('gamma', fontsize=10)
    ax[1][2].set_title('scores (mean & variance) vs gamma, min_alpha=0.05, min_epsilon=0.05', fontsize=10)
    ax[1][2].grid()

    fit_times = []
    mean_scores = []
    var_scores = []
    for min_alpha in alpha_list:
        spec = f'gamma_0.9_min_alpha_{min_alpha}_min_epsilon_0.05'
        spec_res = results[spec]
        Q_track = spec_res['Q_track']
        Q_max = np.max(Q_track, axis=2)
        Q_max = np.max(Q_max, axis=1)
        fit_times.append(spec_res['fit_time'])
        mean_scores.append(np.mean(spec_res['scores']))
        var_scores.append(np.var(spec_res['scores']))
        ax[2][0].plot(Q_max[:Q_plot_range], label=f'min_alpha = {min_alpha}')

    ax[2][0].set_ylabel('max of Q', fontsize=10)
    ax[2][0].set_xlabel(f'num_iterations', fontsize=10)
    ax[2][0].set_title('max of Q vs n_episodes', fontsize=10)
    ax[2][0].legend()
    ax[2][0].grid()

    ax[2][1].plot(alpha_list_plot, fit_times)
    ax[2][1].set_ylabel('fit_time(s)', fontsize=10)
    ax[2][1].set_xlabel('min_alpha', fontsize=10)
    ax[2][1].set_title('fit_time vs min_alpha, gamma=0.1, min_epsilon=0.1', fontsize=10)
    ax[2][1].grid()

    ax_dup = ax[2][2].twinx()
    ax[2][2].plot(alpha_list_plot, mean_scores, label='mean_scores', color='r', marker='o',)
    ax[2][2].set_ylabel('scores (mean)', fontsize=10, color='r')
    ax_dup.plot(alpha_list_plot, var_scores, label='var_scores', color='b', marker='x')
    ax_dup.set_ylabel('scores (variance)', fontsize=10, color='b')
    ax[2][2].set_xlabel('min_alpha', fontsize=10)
    ax[2][2].set_title('scores (mean & variance) vs min_alpha, gamma=0.1, min_epsilon=0.1', fontsize=10)
    ax[2][2].grid()

    fig.savefig(
        os.path.join(BASE_DIR, 'plots', "frozen_lake_q_learning_plot.png"),
        bbox_inches='tight',
        transparent=False
    )


def compare_optimal_policy(env_name: str):
    try:
        assert env_name in ['black_jack', 'frozen_lake']
    except ValueError:
        print("env_name has to be in ['black_jack', 'frozen_lake']")
        raise

    with open(os.path.join(BASE_DIR, 'results', f'{env_name}_vi.pkl'), 'rb') as file:
        vi_res = pickle.load(file)
    with open(os.path.join(BASE_DIR, 'results', f'{env_name}_pi.pkl'), 'rb') as file:
        pi_res = pickle.load(file)
    with open(os.path.join(BASE_DIR, 'results', f'{env_name}_q_learning.pkl'), 'rb') as file:
        q_learning_res = pickle.load(file)

    optimal_vi = get_optimal_policy(vi_res)
    optimal_pi = get_optimal_policy(pi_res)
    optimal_ql = get_optimal_policy(q_learning_res)

    if env_name == 'black_jack':
        optimal_vi = np.reshape(optimal_vi, (29, 10))
        optimal_pi = np.reshape(optimal_pi, (29, 10))
        optimal_ql = np.reshape(optimal_ql, (29, 10))
        fig, ax = plt.subplots(1, 3, figsize=(14, 8))
    else:
        optimal_vi = np.reshape(optimal_vi, (25, 25))
        optimal_pi = np.reshape(optimal_pi, (25, 25))
        optimal_ql = np.reshape(optimal_ql, (25, 25))
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    g1 = sns.heatmap(optimal_vi, ax=ax[0])
    g1.set_ylabel('x')
    g1.set_title(f'{env_name} optimal strategy vi', fontsize=10)

    g2 = sns.heatmap(optimal_pi, ax=ax[1])
    g2.set_title(f'{env_name} optimal strategy pi', fontsize=10)

    g3 = sns.heatmap(optimal_ql, ax=ax[2])
    g3.set_title(f'{env_name} optimal strategy q_learning', fontsize=10)

    fig.savefig(
        os.path.join(BASE_DIR, 'plots', f"{env_name}_heatmap.png"),
        bbox_inches='tight',
        transparent=False
    )
