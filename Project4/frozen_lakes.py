import os
import time
import pickle
import gym
import numpy as np

from gym.envs.toy_text.frozen_lake import generate_random_map
from algorithms.rl_fl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
from typing import Dict
from utils import helper, plotter
from settings import BASE_DIR


def generate_env(map_size: int = 25, frozen_prob: float = 0.95, seed: int = 500):
    np.random.seed(seed)
    random_map = generate_random_map(size=map_size, p=frozen_prob)
    env = gym.make("FrozenLake-v1", desc=random_map)
    return env


def get_q_learning_test_score(env, n_iters=10, pi=None):
    max_episode_steps = 200

    test_scores = np.full([n_iters], np.nan)
    for i in range(0, n_iters):
        state, info = env.reset()
        done = False
        total_reward = 0
        num_step = 0
        while not done:
            action = pi[state]
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward = reward + total_reward
            num_step += 1
            if num_step == max_episode_steps:
                done = 1
        test_scores[i] = total_reward
    env.close()

    return test_scores


def value_iteration_run(max_iter: int = 500) -> Dict:
    frozen_lake = generate_env()
    out = {}
    for theta_pow in range(2, 11, 1):
        theta = 10**(-theta_pow)
        for gamma in np.arange(0.1, 1.0, 0.1):
            gamma = np.round(gamma, 2)
            res = {}
            spec = f'gamma={gamma}_theta={theta}_max_iter={max_iter}'
            print(spec)
            frozen_lake.reset(seed=100)
            st = time.time()
            V, V_track, pi = Planner(frozen_lake.env.P).value_iteration(
                gamma=gamma,
                n_iters=max_iter,
                theta=theta
            )
            fit_time = time.time() - st
            policy = [pi(i) for i in range(25 * 25)]
            res['num_iter'] = max_iter
            res['V'] = V
            res['V_track'] = V_track
            res['fit_time'] = fit_time
            res['optimal_policy'] = policy
            if len(np.where((V_track == 0).all(axis=1))[0]) == 1:
                # meaning no convergence
                res['num_iter_to_converge'] = None
            else:
                res['num_iter_to_converge'] = np.where((V_track == 0).all(axis=1))[0][1]+1
            test_scores = TestEnv.test_env(env=frozen_lake.env, n_iters=100, render=False, user_input=False, pi=pi)
            res['scores'] = test_scores
            out[spec] = res

    with open(os.path.join(BASE_DIR, 'results', 'frozen_lake_vi.pkl'), 'wb') as file:
        pickle.dump(out, file)

    return out


def policy_iteration_run(max_iter: int = 500) -> Dict:
    frozen_lake = generate_env()
    out = {}
    for theta_pow in range(2, 11, 1):
        theta = 10**(-theta_pow)
        for gamma in np.arange(0.1, 1.0, 0.1):
            gamma = np.round(gamma, 2)
            res = {}
            spec = f'gamma={gamma}_theta={theta}_max_iter={max_iter}'
            print(spec)
            frozen_lake.reset(seed=100)
            st = time.time()
            V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration(
                gamma=gamma,
                n_iters=max_iter,
                theta=theta
            )
            fit_time = time.time() - st
            policy = [pi(i) for i in range(25 * 25)]
            res['num_iter'] = max_iter
            res['V'] = V
            res['V_track'] = V_track
            res['fit_time'] = fit_time
            res['optimal_policy'] = policy
            if len(np.where((V_track == 0).all(axis=1))[0]) == 1:
                # meaning no convergence
                res['num_iter_to_converge'] = None
            else:
                res['num_iter_to_converge'] = np.where((V_track == 0).all(axis=1))[0][1]+1
            test_scores = TestEnv.test_env(env=frozen_lake.env, n_iters=100, render=False, user_input=False, pi=pi)
            res['scores'] = test_scores
            out[spec] = res

    with open(os.path.join(BASE_DIR, 'results', 'frozen_lake_pi.pkl'), 'wb') as file:
        pickle.dump(out, file)

    return out


def q_learning_run(n_episodes: int = 50000) -> Dict:
    frozen_lake = generate_env()
    out = {}
    gamma_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_list = [0.01, 0.05, 0.10]
    epsilon_list = [0.01, 0.05, 0.10]

    for gamma in gamma_list:
        for alpha in alpha_list:
            for epsilon in epsilon_list:
                res = {}
                spec = f'gamma_{gamma}_min_alpha_{alpha}_min_epsilon_{epsilon}'
                print(spec)
                frozen_lake.reset(seed=200)
                st = time.time()
                Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning(
                    gamma=gamma,
                    init_alpha=0.9,
                    min_alpha=alpha,
                    alpha_decay_ratio=0.5,
                    init_epsilon=1.0,
                    min_epsilon=epsilon,
                    epsilon_decay_ratio=0.9,
                    n_episodes=n_episodes
                )
                fit_time = time.time() - st
                policy = [pi(i) for i in range(25 * 25)]
                res['n_episodes'] = n_episodes
                res['Q'] = Q
                res['Q_track'] = Q_track
                res['optimal_policy'] = policy
                res['pi_track'] = Q_track
                res['fit_time'] = fit_time

                test_scores = get_q_learning_test_score(env=frozen_lake.env, n_iters=100, pi=policy)
                res['scores'] = test_scores
                out[spec] = res

    with open(os.path.join(BASE_DIR, 'results', 'frozen_lake_q_learning.pkl'), 'wb') as file:
        pickle.dump(out, file)

    return out


def run_frozen_lake_all():
    value_iteration_run()
    policy_iteration_run()
    q_learning_run()

    pi_df = helper.convert_vi_pi_res_to_df(run_name='frozen_lake_pi')
    vi_df = helper.convert_vi_pi_res_to_df(run_name='frozen_lake_vi')
    ql_df = helper.convert_ql_res_to_df(run_name='frozen_lake_q_learning')

    plotter.plot_vi_pi_convergence(run_name='frozen_lake_vi', var_type='gamma')
    plotter.plot_vi_pi_convergence(run_name='frozen_lake_vi', var_type='theta')
    plotter.plot_vi_pi_convergence(run_name='frozen_lake_pi', var_type='gamma')
    plotter.plot_vi_pi_convergence(run_name='frozen_lake_pi', var_type='theta')
    plotter.plot_frozen_lake_ql_convergence()
    plotter.compare_optimal_policy('frozen_lake')

