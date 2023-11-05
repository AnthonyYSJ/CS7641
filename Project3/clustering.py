import os
import time
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score
from sklearn.manifold import TSNE
from typing import Dict
from preprocess_data import preprocess_credit_card_data, preprocess_customer_segmentation_data
from settings import BASE_DIR


def run_clustering(algo: str = 'EM', dataset_num: int = 1) -> Dict:
    assert algo in ['EM', 'KMeans']
    assert dataset_num in [1, 2]
    if dataset_num == 1:
        X, y = preprocess_credit_card_data()
    else:
        X, y = preprocess_customer_segmentation_data()

    num_true_cluster = len(set(list(y.reshape(-1, ))))
    num_clusters = list(np.arange(2, 11, 1))
    sil_scores = []
    fit_times = []
    best_sil = -np.inf
    best_c = 0
    best_model = None
    f1_score_ = None
    true_res = {}

    for c in num_clusters:
        if algo == 'EM':
            clusterer = GaussianMixture(n_components=c, covariance_type='diag',
                                        n_init=1, warm_start=True, random_state=100)
        else:  # algo == 'KMeans"
            clusterer = KMeans(n_clusters=c, init='k-means++', n_init=10, random_state=100)
        st = time.time()
        clusterer.fit(X)
        fit_times.append(time.time() - st)

        labels = clusterer.predict(X)
        sil_score = silhouette_score(X, labels)
        sil_scores.append(sil_score)
        if sil_score > best_sil:
            best_sil = sil_score
            best_c = c
            best_model = copy.deepcopy(clusterer)

        if c == num_true_cluster:
            if c != 2:
                f1_score_ = f1_score(y, labels, average='weighted')
            else:
                f1_score_ = f1_score(y, labels)
            true_res = {
                'sil_score': sil_score,
                'c': c,
                'model': copy.deepcopy(clusterer),
            }

    best_res = {
        'best_sil': best_sil,
        'best_c': best_c,
        'best_model': best_model,
    }

    X_tsne = TSNE(n_components=2).fit_transform(X)
    df_tsne = pd.DataFrame(X_tsne)
    df_tsne['cluster'] = true_res['model'].predict(X)
    df_tsne.columns = ['x1', 'x2', 'cluster']

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    ax[0].plot(num_clusters, sil_scores, marker='o')
    ax[0].set_ylabel('silhouette score', fontsize=10)
    ax[0].set_xlabel(f'num_clusters size', fontsize=10)
    ax[0].set_title('silhouette score vs num_clusters', fontsize=10)
    ax[0].grid()

    ax[1].plot(num_clusters, fit_times, marker='o')
    ax[1].set_ylabel('fit_time', fontsize=10)
    ax[1].set_xlabel(f'num_clusters size', fontsize=10)
    ax[1].set_title('fit_time vs num_clusters', fontsize=10)
    ax[1].grid()

    sns.scatterplot(data=df_tsne, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[2])
    ax[2].set_title('Visualize cluster results on T-SNE 2D', fontsize=10)
    fig.suptitle(f'Dataset {dataset_num}: Cluster Results For {algo}')
    fig.savefig(os.path.join(BASE_DIR, 'plots', 'clustering', f'{algo}_clustering_{dataset_num}.png'),
                bbox_inches='tight', transparent=False)
    plt.close()

    res = {
        'dataset_num': dataset_num, 'num_clusters': num_clusters, 'sil_scores': sil_scores, 'fit_times': fit_times,
        'f1_score': f1_score_, 'best_model_res': best_res, 'true_label_model_res': true_res,
    }

    return res
