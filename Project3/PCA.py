import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict
from preprocess_data import preprocess_credit_card_data, preprocess_customer_segmentation_data
from settings import BASE_DIR


def run_and_plot_pca(dataset_num: int = 1) -> Dict:
    assert dataset_num in [1, 2]
    if dataset_num == 1:
        X, y = preprocess_credit_card_data()
    else:
        X, y = preprocess_customer_segmentation_data()

    vif_original = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    pca = PCA(random_state=1)
    n_components = [i for i in range(1, X.shape[1]+1)]
    losses = []
    fit_times = []

    for i in n_components:
        params = {'n_components': i}
        pca.set_params(**params)
        st = time.time()
        pca.fit(X)
        fit_times.append(time.time()-st)
        X_transformed = pca.transform(X)
        X_projected = pca.inverse_transform(X_transformed)
        # MSE Error
        loss = np.sum((X - X_projected) ** 2, axis=1).mean()
        losses.append(loss)

    vif_transformed = [variance_inflation_factor(X_transformed, i) for i in range(X_transformed.shape[1])]
    explained_variances = pca.explained_variance_
    cumulative_explained_variance_ratios = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].plot(n_components, explained_variances, marker='o')
    ax[0].set_ylabel('eigenvalues', fontsize=10)
    ax[0].set_xlabel(f'n_component', fontsize=10)
    ax[0].set_title('eigenvalues vs n_component', fontsize=10)
    ax[0].grid()

    ax[1].plot(n_components, cumulative_explained_variance_ratios, marker='o')
    ax[1].set_ylabel('cumulative explained_variance_ratio', fontsize=10)
    ax[1].set_xlabel(f'n_components', fontsize=10)
    ax[1].set_title('cumulative explained_variance_ratio vs n_components', fontsize=10)
    ax[1].grid()

    ax[2].plot(n_components, losses, marker='o')
    ax[2].set_ylabel('reconstruction error', fontsize=10)
    ax[2].set_xlabel(f'n_components', fontsize=10)
    ax[2].set_title('reconstruction error vs n_components', fontsize=10)
    ax[2].grid()

    fig.suptitle(f'Dataset {dataset_num}: PCA Results')
    fig.savefig(os.path.join(BASE_DIR, 'plots', 'dim_reduction',  f'PCA_{dataset_num}.png'),
                bbox_inches='tight', transparent=False)
    plt.close()

    return {'dataset_num': dataset_num, 'n_components': n_components, 'error': losses,
            'explained_variances': explained_variances,
            'cumulative_explained_variance_ratios': cumulative_explained_variance_ratios,
            'fit_time': fit_times,
            'vif_original': vif_original,
            'vif_transformed': vif_transformed}
