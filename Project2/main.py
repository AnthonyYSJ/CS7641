from four_peaks import run_four_peaks
from knapsack import run_knapsack
from flipflop import run_flipflop
from one_max import run_one_max
from neural_net import run_nn


def run_part1():
    run_four_peaks()
    run_knapsack()
    run_flipflop()
    run_one_max()


def run_part2():
    run_nn()


if __name__ == "__main__":
    run_part1()
    run_part2()

