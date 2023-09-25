from decision_tree import run_decision_trees
from adaBoost_tree import run_adaBoost_trees
from neural_network import run_neural_network
from svc import run_svc
from knn import run_knn


if __name__ == "__main__":
    run_decision_trees()
    run_adaBoost_trees()
    run_neural_network()
    run_svc()
    run_knn()

    print("Done!")
