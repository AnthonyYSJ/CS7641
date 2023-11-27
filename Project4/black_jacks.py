import os
import time
import pickle
import gym
import numpy as np

from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
from typing import Dict
from utils import helper, plotter
from settings import BASE_DIR


class Blackjack:
    def __init__(self):
        np.random.seed(500)
        self._env = gym.make('Blackjack-v1', render_mode=None)
        # Explanation of convert_state_obs lambda:
        # def function(state, done):
        #     if done:
        #         return -1
        #    else:
        #         if state[2]:
        #             int(f"{state[0]+6}{(state[1]-2)%10}")
        #         else:
        #             int(f"{state[0]-4}{(state[1]-2)%10}")
        self._convert_state_obs = lambda state, done: (
            -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
                f"{state[0] - 4}{(state[1] - 2) % 10}"))
        # Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
        current_dir = os.path.dirname(__file__)
        file_name = 'blackjack-envP'
        f = os.path.join(current_dir, file_name)
        try:
            self._P = pickle.load(open(f, "rb"))
        except IOError:
            print("Pickle load failed.  Check path", f)
        self._n_actions = self.env.action_space.n
        self._n_states = len(self._P)

    @property
    def n_actions(self):
        return self._n_actions

    @n_actions.setter
    def n_actions(self, n_actions):
        self._n_actions = n_actions

    @property
    def n_states(self):
        return self._n_states

    @n_states.setter
    def n_states(self, n_states):
        self._n_states = n_states

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    @property
    def convert_state_obs(self):
        return self._convert_state_obs

    @convert_state_obs.setter
    def convert_state_obs(self, convert_state_obs):
        self._convert_state_obs = convert_state_obs


def value_iteration_run(max_iter: int = 500) -> Dict:
    out = {}
    for theta_pow in range(2, 11, 1):
        theta = 10**(-theta_pow)
        for gamma in np.arange(0.1, 1.0, 0.1):
            gamma = np.round(gamma, 2)
            res = {}
            spec = f'gamma={gamma}_theta={theta}_max_iter={max_iter}'
            print(spec)
            black_jack = Blackjack()
            st = time.time()
            V, V_track, pi = Planner(black_jack.P).value_iteration(
                gamma=gamma,
                n_iters=max_iter,
                theta=theta
            )
            fit_time = time.time() - st
            policy = [pi(i) for i in range(290)]
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
            test_scores = TestEnv.test_env(
                env=black_jack.env,
                n_iters=100,
                render=False,
                user_input=False,
                convert_state_obs=black_jack.convert_state_obs,
                pi=pi
            )
            res['scores'] = test_scores
            out[spec] = res

    with open(os.path.join(BASE_DIR, 'results', 'black_jack_vi.pkl'), 'wb') as file:
        pickle.dump(out, file)

    return out


def policy_iteration_run(max_iter: int = 500) -> Dict:
    black_jack = Blackjack()
    out = {}
    for theta_pow in range(2, 11, 1):
        theta = 10**(-theta_pow)
        for gamma in np.arange(0.1, 1.0, 0.1):
            gamma = np.round(gamma, 2)
            res = {}
            spec = f'gamma={gamma}_theta={theta}_max_iter={max_iter}'
            print(spec)
            st = time.time()
            V, V_track, pi = Planner(black_jack.P).policy_iteration(
                gamma=gamma,
                n_iters=max_iter,
                theta=theta
            )
            fit_time = time.time() - st
            policy = [pi(i) for i in range(290)]
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
            test_scores = TestEnv.test_env(
                env=black_jack.env,
                n_iters=100,
                render=False,
                user_input=False,
                convert_state_obs=black_jack.convert_state_obs,
                pi=pi
            )
            res['scores'] = test_scores
            out[spec] = res
    with open(os.path.join(BASE_DIR, 'results', 'black_jack_pi.pkl'), 'wb') as file:
        pickle.dump(out, file)

    return out


def q_learning_run(n_episodes: int = 50000) -> Dict:
    black_jack = Blackjack()
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
                st = time.time()
                Q, V, pi, Q_track, pi_track = RL(black_jack.env).q_learning(
                    nS=black_jack.n_states,
                    nA=black_jack.n_actions,
                    convert_state_obs=black_jack.convert_state_obs,
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
                policy = [pi(i) for i in range(290)]
                res['n_episodes'] = n_episodes
                res['Q'] = Q
                res['Q_track'] = Q_track
                res['optimal_policy'] = policy
                res['pi_track'] = Q_track
                res['fit_time'] = fit_time
                test_scores = TestEnv.test_env(
                    env=black_jack.env,
                    n_iters=100,
                    render=False,
                    user_input=False,
                    convert_state_obs=black_jack.convert_state_obs,
                    pi=pi
                )
                res['scores'] = test_scores
                out[spec] = res

    with open(os.path.join(BASE_DIR, 'results', 'black_jack_q_learning.pkl'), 'wb') as file:
        pickle.dump(out, file)

    return out


def run_black_jack_all():
    vi_res = value_iteration_run()
    pi_res = policy_iteration_run()
    q_learning_res = q_learning_run()

    pi_df = helper.convert_vi_pi_res_to_df(run_name='black_jack_pi')
    vi_df = helper.convert_vi_pi_res_to_df(run_name='black_jack_vi')
    ql_df = helper.convert_ql_res_to_df(run_name='black_jack_q_learning')

    plotter.plot_vi_pi_convergence(run_name='black_jack_vi', var_type='gamma')
    plotter.plot_vi_pi_convergence(run_name='black_jack_vi', var_type='theta')
    plotter.plot_vi_pi_convergence(run_name='black_jack_pi', var_type='gamma')
    plotter.plot_vi_pi_convergence(run_name='black_jack_pi', var_type='theta')
    plotter.plot_black_jack_ql_convergence()
    plotter.compare_optimal_policy('black_jack')
