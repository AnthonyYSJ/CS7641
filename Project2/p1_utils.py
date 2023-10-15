import os
import time
import pickle
import copy
import pathlib
import numpy as np
import mlrose_hiive as mlrose
from settings import BASE_DIR


def run_p1_algo(problem_type, algo: str, problem_size: int):
    if problem_type == 'four_peaks':
        fitness = mlrose.FourPeaks(t_pct=.5)
        problem_fit = mlrose.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True, max_val=2)
    elif problem_type == 'knapsack':
        weights = np.random.uniform(low=0.1, high=1, size=(problem_size,))
        values = np.random.uniform(low=1, high=problem_size, size=(problem_size,))
        fitness = mlrose.Knapsack(weights, values)
        problem_fit = mlrose.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True, max_val=2)
    elif problem_type == 'flipflop':
        fitness = mlrose.FlipFlop()
        problem_fit = mlrose.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True, max_val=2)
    else:  # problem_type = "one_max"
        fitness = mlrose.OneMax()
        problem_fit = mlrose.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True, max_val=2)

    if algo == 'RHC':
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(
            problem_fit, restarts=10 * problem_size, max_attempts=10,
            max_iters=problem_size * 10, init_state=None, curve=True
        )
    elif algo == 'GA':
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(
            problem_fit, pop_size=10 * problem_size, mutation_prob=0.4,
            max_attempts=10, max_iters=problem_size * 10, curve=True
        )
    elif algo == 'SA':
        if problem_type == 'one_max':
            best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
                problem_fit, schedule=mlrose.GeomDecay(), max_attempts=100,
                init_state=None, max_iters=problem_size * 10, curve=True
            )
        else:
            best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
                problem_fit, schedule=mlrose.GeomDecay(), max_attempts=10,
                init_state=None, max_iters=problem_size * 10, curve=True
            )
    else:  # algo == 'MIMIC'
        best_state, best_fitness, fitness_curve = mlrose.mimic(
            problem_fit, pop_size=10 * problem_size, keep_pct=0.2,
            max_attempts=10, max_iters=problem_size * 10, curve=True
        )

    return best_fitness, fitness_curve


def get_p1_res(problem_type, run_from_scratch: bool = False, save_res: bool = True, verbose: bool = False):
    print(f'problem_type = {problem_type}')

    res_file = pathlib.Path(os.path.join(BASE_DIR, 'results', f'{problem_type}_run_results.pkl'))

    if not res_file.is_file():
        if not run_from_scratch:
            raise ValueError('run_from_scratch = False, but there is no existing result! Set run_from_scratch = True')
        else:
            print('run starts...')

    if res_file.is_file():
        if not run_from_scratch:
            with open(os.path.join(BASE_DIR, 'results', f'{problem_type}_run_results.pkl'), 'rb') as file:
                res = pickle.load(file)
            print('results already exists, loading saved result...')
            return res
        else:
            print('results already exists, since run_from_scratch=True, starts run and overwrite previous result...')

    algorithms = ['RHC', 'GA', 'SA', 'MIMIC']
    result_types = ['size', 'time', 'best_fitness', 'fitness_curve']

    result_list = {}
    for r_type in result_types:
        result_list[r_type] = []

    res = {}
    for algorithm in algorithms:
        res[algorithm] = copy.deepcopy(result_list)

    for size in range(10, 100, 10):
        for algorithm in algorithms:
            if verbose:
                print(f'size = {size}, algorithm = {algorithm}')
            st_time = time.time()
            best_fitness, fitness_curve = run_p1_algo(problem_type, algo=algorithm, problem_size=size)
            et_time = time.time()
            res[algorithm]['size'].append(size)
            res[algorithm]['time'].append(et_time - st_time)
            res[algorithm]['best_fitness'].append(best_fitness)
            res[algorithm]['fitness_curve'].append(fitness_curve)

    if save_res:
        with open(os.path.join(BASE_DIR, 'results', f'{problem_type}_run_results.pkl'), 'wb') as file:
            pickle.dump(res, file)

    return res
