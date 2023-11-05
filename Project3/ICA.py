import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from typing import Dict
from preprocess_data import preprocess_credit_card_data, preprocess_customer_segmentation_data
from settings import BASE_DIR


def run_and_plot_ica(dataset_num: int = 1) -> Dict:
    assert dataset_num in [1, 2]
    if dataset_num == 1:
        X, y = preprocess_credit_card_data()
    else:
        X, y = preprocess_customer_segmentation_data()

    ica = FastICA(random_state=10)
    n_components = [i for i in range(1, X.shape[1])]
    kurtosis_list = []
    losses = []
    fit_times = []

    for i in n_components:
        params = {'n_components': i}
        ica.set_params(**params)
        st = time.time()
        ica.fit(X)
        fit_times.append(time.time()-st)
        X_transformed = ica.transform(X)
        kurtosis_list.append(np.mean(np.abs(kurtosis(X_transformed))))
        X_projected = ica.inverse_transform(X_transformed)
        # MSE Error
        loss = np.sum((X - X_projected) ** 2, axis=1).mean()
        losses.append(loss)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(n_components, kurtosis_list, marker='o')
    ax[0].set_ylabel('average kurtosis', fontsize=10)
    ax[0].set_xlabel(f'n_component', fontsize=10)
    ax[0].set_title('average kurtosis vs n_component', fontsize=10)
    ax[0].grid()

    ax[1].plot(n_components, losses, marker='o')
    ax[1].set_ylabel('reconstruction error', fontsize=10)
    ax[1].set_xlabel(f'n_components', fontsize=10)
    ax[1].set_title('reconstruction error vs n_components', fontsize=10)
    ax[1].grid()

    fig.suptitle(f'Dataset {dataset_num}: ICA Results')
    fig.savefig(os.path.join(BASE_DIR, 'plots', 'dim_reduction', f'ICA_{dataset_num}.png'),
                bbox_inches='tight', transparent=False)
    plt.close()

    return {'dataset_num': dataset_num, 'n_components': n_components, 'error': losses,
            'kurtosis_list': kurtosis_list, 'fit_time': fit_times}
