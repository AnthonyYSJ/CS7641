from utils import get_p1_res
from plotter import plot_res


def run_knapsack():
    problem_type = 'knapsack'
    res = get_p1_res(problem_type=problem_type, run_from_scratch=True, save_res=True, verbose=True)
    plot_res(res, problem_type)