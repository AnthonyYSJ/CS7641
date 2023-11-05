from clustering import run_clustering
from PCA import run_and_plot_pca
from ICA import run_and_plot_ica
from RCA import run_and_plot_rca
from Manifold import run_and_plot_manifold
from dim_red_cluster import run_all_dim_red_cluster
from neural_network import run_nn


def run_part1():
    run_clustering(algo='EM', dataset_num=1)
    run_clustering(algo='KMeans', dataset_num=1)
    run_clustering(algo='EM', dataset_num=2)
    run_clustering(algo='KMeans', dataset_num=2)


def run_part2():
    # PCA
    run_and_plot_pca(dataset_num=1)
    run_and_plot_pca(dataset_num=2)

    # ICA
    run_and_plot_ica(dataset_num=1)
    run_and_plot_ica(dataset_num=2)

    # RCA
    run_and_plot_rca(dataset_num=1)
    run_and_plot_rca(dataset_num=2)

    # Manifold
    run_and_plot_manifold(dataset_num=1)
    run_and_plot_manifold(dataset_num=2)


def run_part3():
    run_all_dim_red_cluster()


def run_part4():
    run_nn()


if __name__ == "__main__":
    run_part1()
    run_part2()
    run_part3()
    run_part4()
