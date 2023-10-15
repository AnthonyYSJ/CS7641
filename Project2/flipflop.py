from p1_utils import get_p1_res
from plotter import plot_p1_res


def run_flipflop():
    problem_type = 'flipflop'
    res = get_p1_res(problem_type=problem_type, run_from_scratch=True, save_res=True, verbose=True)
    plot_p1_res(res, problem_type)
