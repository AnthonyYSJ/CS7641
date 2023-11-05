import os
import time
import matplotlib.pyplot as plt

from sklearn.manifold import Isomap

from typing import Dict
from preprocess_data import preprocess_credit_card_data, preprocess_customer_segmentation_data
from settings import BASE_DIR


def run_and_plot_manifold(dataset_num: int = 1) -> Dict:
    assert dataset_num in [1, 2]
    if dataset_num == 1:
        X, y = preprocess_credit_card_data()
    else:
        X, y = preprocess_customer_segmentation_data()

    iso = Isomap()
    n_components = [i for i in range(1, X.shape[1] + 1)]
    losses = []
    fit_times = []

    for i in n_components:
        print(i)
        params = {'n_components': i}
        iso.set_params(**params)
        st = time.time()
        iso.fit(X)
        fit_times.append(time.time() - st)
        # Reconstruction Error
        loss = iso.reconstruction_error()
        losses.append(loss)

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax[0].plot(n_components, losses, marker='o')
    ax[0].set_ylabel('reconstruction error', fontsize=10)
    ax[0].set_xlabel(f'n_components', fontsize=10)
    ax[0].set_title('reconstruction error vs n_components', fontsize=10)
    ax[0].grid()

    ax[1].plot(n_components, fit_times, marker='o')
    ax[1].set_ylabel('fit_time', fontsize=10)
    ax[1].set_xlabel(f'n_component', fontsize=10)
    ax[1].set_title('fit_time vs n_component', fontsize=10)
    ax[1].grid()

    fig.suptitle(f'Dataset {dataset_num}: ISO Results')
    fig.savefig(os.path.join(BASE_DIR, 'plots', 'dim_reduction', f'ISO_{dataset_num}.png'),
                bbox_inches='tight', transparent=False)
    plt.close()

    return {'dataset_num': dataset_num, 'n_components': n_components, 'error': losses, 'fit_time': fit_times}
