import os
import time
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score, f1_score
from sklearn.manifold import TSNE
from typing import Dict
from preprocess_data import preprocess_credit_card_data, preprocess_customer_segmentation_data
from settings import BASE_DIR


def run_clustering_with_dim_reduction(cl_algo: str = 'EM', dr_algo: str = 'PCA', dataset_num: int = 1) -> Dict:
    assert cl_algo in ['EM', 'KMeans']
    assert dr_algo in ['PCA', 'ICA', 'RCA', 'ISO']
    assert dataset_num in [1, 2]

    if dataset_num == 1:
        X, y = preprocess_credit_card_data()
        if dr_algo == 'PCA':
            dr = PCA(n_components=12, random_state=1)
        elif dr_algo == 'ICA':
            dr = FastICA(n_components=6, random_state=10)
        elif dr_algo == 'RCA':
            dr = SparseRandomProjection(n_components=16, random_state=1)
        else:  # dr_algo == 'ISO'
            dr = Isomap(n_components=9)

        num_true_cluster = len(set(list(y.reshape(-1, ))))
        original_clusterer = GaussianMixture(n_components=num_true_cluster, covariance_type='diag',
                                             n_init=1, warm_start=True, random_state=100)
    else:  # dataset_num == 1
        X, y = preprocess_customer_segmentation_data()
        if dr_algo == 'PCA':
            dr = PCA(n_components=6, random_state=1)
        elif dr_algo == 'ICA':
            dr = FastICA(n_components=4, random_state=10)
        elif dr_algo == 'RCA':
            dr = SparseRandomProjection(n_components=5, random_state=1)
        else:  # dr_algo == 'ISO'
            dr = Isomap(n_components=6)

        num_true_cluster = len(set(list(y.reshape(-1, ))))
        original_clusterer = KMeans(n_clusters=num_true_cluster, init='k-means++', n_init=10, random_state=100)

    original_clusterer.fit(X)
    original_labels = original_clusterer.predict(X)

    X_transformed = dr.fit_transform(X)

    num_clusters = list(np.arange(2, 11, 1))
    sil_scores = []
    fit_times = []
    best_sil = -np.inf
    best_c = 0
    best_model = None
    f1_score_ = None
    true_res = {}

    for c in num_clusters:
        if cl_algo == 'EM':
            clusterer = GaussianMixture(n_components=c, covariance_type='diag',
                                        n_init=1, warm_start=True, random_state=100)
        else:  # cl_algo == 'KMeans"
            clusterer = KMeans(n_clusters=c, init='k-means++', n_init=10, random_state=100)
        st = time.time()
        clusterer.fit(X_transformed)
        fit_times.append(time.time() - st)

        labels = clusterer.predict(X_transformed)
        sil_score = silhouette_score(X_transformed, labels)
        sil_scores.append(sil_score)
        if sil_score > best_sil:
            best_sil = sil_score
            best_c = c
            best_model = copy.deepcopy(clusterer)

        if c == num_true_cluster:
            after_labels = labels
            if c != 2:
                f1_score_ = f1_score(y, labels, average='weighted')
            else:
                f1_score_ = f1_score(y, labels)
            true_res = {
                'sil_score': sil_score,
                'c': c,
                'model': copy.deepcopy(clusterer),
            }

    match_percentage = (original_labels == after_labels).sum()/len(original_labels)

    best_res = {
        'best_sil': best_sil,
        'best_c': best_c,
        'best_model': best_model,
    }

    X_tsne = TSNE(n_components=2).fit_transform(X_transformed)
    df_tsne = pd.DataFrame(X_tsne)
    df_tsne['cluster'] = true_res['model'].predict(X_transformed)
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
    fig.suptitle(f'Dataset {dataset_num}: Results For {cl_algo} + {dr_algo}')
    fig.savefig(os.path.join(BASE_DIR, 'plots', 'dim_red_clustering', f'{cl_algo}_{dr_algo}_{dataset_num}.png'),
                bbox_inches='tight', transparent=False)
    plt.close()

    res = {
        'num_clusters': num_clusters, 'sil_scores': sil_scores, 'fit_times': fit_times,
        'f1_score': f1_score_, 'best_model_res': best_res, 'true_label_model_res': true_res,
        'match_percentage': match_percentage,
    }

    return res


def run_all_dim_red_cluster(save_res: bool = True, verbose: bool = True) -> Dict:
    all_res = {}
    for dataset_num in [1, 2]:
        for cl_algo in ['EM', 'KMeans']:
            for dr_algo in ['PCA', 'ICA', 'RCA', 'ISO']:
                if verbose:
                    print(f'dataset_num = {dataset_num}, clustering algorithm = {cl_algo}, '
                          f'dimension reduction algorithm = {dr_algo}')
                res = run_clustering_with_dim_reduction(cl_algo, dr_algo, dataset_num)
                if verbose:
                    print(f"f1_score = {res['f1_score']}")
                    all_res[f'{dr_algo}_{cl_algo}_{dataset_num}'] = res
    if save_res:
        with open(os.path.join(BASE_DIR, 'results', 'all_dim_reduction_clustering_res.pkl'), 'wb') as file:
            pickle.dump(all_res, file)


    with open(os.path.join(BASE_DIR, 'results', 'all_dim_reduction_clustering_res.pkl'), 'rb') as file:
        all_res = pickle.load(file)

    algo_specs = []
    best_sils = []
    f1_scores = []
    match_ps = []

    for algo_sepc in all_res:
        algo_specs.append(algo_sepc)
        best_sils.append(all_res[algo_sepc]['best_model_res']['best_sil'])
        f1_scores.append(all_res[algo_sepc]['f1_score'])
        match_ps.append(all_res[algo_sepc]['match_percentage'])

    out = pd.DataFrame({'algo_specs': algo_specs, 'best_silhouette_score': best_sils,
                        'f1_score': f1_scores, 'match_percentage': match_ps})

    out.to_csv(os.path.join(BASE_DIR, 'results', 'dim_red_cluster_results.csv'), index=False)
