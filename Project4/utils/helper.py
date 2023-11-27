import os
import pickle
import numpy as np
import pandas as pd

from typing import Dict, List
from settings import BASE_DIR


def convert_vi_pi_res_to_df(run_name: str) -> pd.DataFrame:
    with open(os.path.join(BASE_DIR, 'results', f'{run_name}.pkl'), 'rb') as file:
        results = pickle.load(file)

    df = {
        'theta': [],
        'gamma': [],
        'mean_scores': [],
        'var_scores': [],
        'num_iter_to_converge': [],
        'highest_Vs': [],
    }

    for theta_pow in range(2, 11, 1):
        theta = 10 ** (-theta_pow)
        for gamma in np.arange(0.1, 1.0, 0.1):
            gamma = np.round(gamma, 2)
            spec = f'gamma={gamma}_theta={theta}_max_iter=500'
            spec_res = results[spec]
            df['theta'].append(theta)
            df['gamma'].append(gamma)
            df['mean_scores'].append(np.mean(spec_res['scores']))
            df['var_scores'].append(np.var(spec_res['scores']))
            df['num_iter_to_converge'].append((spec_res['num_iter_to_converge']))
            df['highest_Vs'].append((np.max(spec_res['V'])))

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(BASE_DIR, 'results', f'{run_name}_results.csv'), index=False)

    return df


def convert_ql_res_to_df(run_name: str) -> pd.DataFrame:
    with open(os.path.join(BASE_DIR, 'results', f'{run_name}.pkl'), 'rb') as file:
        results = pickle.load(file)

    df = {
        'gamma': [],
        'min_alpha': [],
        'min_epsilon': [],
        'mean_scores': [],
        'var_scores': [],
        'highest_Q': [],
    }

    gamma_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_list = [0.01, 0.05, 0.10]
    epsilon_list = [0.01, 0.05, 0.10]

    for gamma in gamma_list:
        for min_alpha in alpha_list:
            for min_epsilon in epsilon_list:
                spec = f'gamma_{gamma}_min_alpha_{min_alpha}_min_epsilon_{min_epsilon}'
                spec_res = results[spec]
                df['gamma'].append(gamma)
                df['min_alpha'].append(min_alpha)
                df['min_epsilon'].append(min_epsilon)
                df['mean_scores'].append(np.mean(spec_res['scores']))
                df['var_scores'].append(np.var(spec_res['scores']))
                df['highest_Q'].append((np.max(spec_res['Q'])))

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(BASE_DIR, 'results', f'{run_name}_results.csv'), index=False)

    return df


def get_optimal_policy(results: Dict) -> List:
    max_score = -np.inf
    optimal_policy = None
    for spec in results:
        spec_res = results[spec]
        mean_score = np.mean(spec_res['scores'])
        if mean_score > max_score:
            max_score = mean_score
            optimal_policy = spec_res['optimal_policy']
    return optimal_policy
