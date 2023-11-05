import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import mean_squared_error
from typing import Dict
from preprocess_data import preprocess_credit_card_data, preprocess_customer_segmentation_data
from settings import BASE_DIR


def run_and_plot_rca(dataset_num: int = 1) -> Dict:
    assert dataset_num in [1, 2]
    if dataset_num == 1:
        X, y = preprocess_credit_card_data()
    else:
        X, y = preprocess_customer_segmentation_data()

    rca = SparseRandomProjection(random_state=1)
    n_components = [i for i in range(1, X.shape[1]+1)]
    losses = []
    std_losses = []
    fit_times = []

    for i in n_components:
        tmp = []
        for j in range(1, 11):
            params = {'n_components': i, 'random_state': j}
            rca.set_params(**params)
            st = time.time()
            rca.fit(X)
            fit_times.append(time.time()-st)
            components = rca.components_.toarray()
            X_inverse = np.linalg.pinv(components.T)
            X_transformed = rca.transform(X)
            X_projected = X_transformed.dot(X_inverse)
            loss = mean_squared_error(X, X_projected)
            tmp.append(loss)
        losses.append(np.mean(tmp))
        std_losses.append(np.std(tmp))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(n_components, losses, marker='o')
    ax[0].set_ylabel('reconstruction error', fontsize=10)
    ax[0].set_xlabel(f'n_components', fontsize=10)
    ax[0].set_title('average of reconstruction error vs n_components', fontsize=10)
    ax[0].grid()

    ax[1].plot(n_components, std_losses, marker='o')
    ax[1].set_ylabel('variance of reconstruction error', fontsize=10)
    ax[1].set_xlabel(f'n_components', fontsize=10)
    ax[1].set_title('variance of reconstruction error vs n_components', fontsize=10)
    ax[1].grid()

    fig.suptitle(f'Dataset {dataset_num}: RCA Results')
    fig.savefig(os.path.join(BASE_DIR, 'plots', 'dim_reduction', f'RCA_{dataset_num}.png'),
                bbox_inches='tight', transparent=False)
    plt.close()

    return {'dataset_num': dataset_num, 'n_components': n_components, 'error': losses,
            'std_losses': std_losses, 'fit_time': fit_times}
