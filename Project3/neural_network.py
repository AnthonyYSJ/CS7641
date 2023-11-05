import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import pandas as pd
import copy
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import Isomap
from typing import Dict
from preprocess_data import preprocess_credit_card_data, normal_train_test_split
from settings import BASE_DIR


class CustomNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers=None, output_dim: int = 1):
        super().__init__()
        self.num_layers = len(hidden_layers)

        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.batch_norm_input = nn.BatchNorm1d(hidden_layers[0])

        if len(hidden_layers) >= 2:
            self.fc1 = nn.Linear(hidden_layers[0], hidden_layers[1])
            self.batch_norm_1 = nn.BatchNorm1d(hidden_layers[1])
        if len(hidden_layers) >= 3:
            self.fc2 = nn.Linear(hidden_layers[1], hidden_layers[2])
            self.batch_norm_2 = nn.BatchNorm1d(hidden_layers[2])

        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.batch_norm_input(x)
        x = self.relu(x)

        if self.num_layers >= 2:
            x = self.fc1(x)
            x = self.batch_norm_1(x)
            x = self.relu(x)
        if self.num_layers >= 3:
            x = self.fc2(x)
            x = self.batch_norm_2(x)
            x = self.relu(x)

        output = self.sigmoid(self.output_layer(x))
        return output


def model_train(X_train, y_train, X_test, y_test,
                model_params=None, n_epochs=100, lr=1e-5, batch_size=64,
                dr_algo: str = 'No_Dim_Reduction', cl_algo: str = 'No_Clustering') -> Dict:
    if model_params is None:
        model_params = [512, 64, 16]
    # loss function and optimizer
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CustomNet(input_dim=X_train.shape[1], hidden_layers=model_params)
    model.to(device)

    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_f1 = - np.inf
    train_acc_best = - np.inf
    train_recall_best = - np.inf
    test_acc_best = -np.inf
    test_recall_best = -np.inf
    best_weights = None

    st = time.time()
    res = {
        'train_acc': [],
        'train_recall': [],
        'test_acc': [],
        'test_recall': [],
        'train_loss': [],
    }

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        model.train()
        train_acc_batch_list = []
        train_recall_batch_list = []
        train_loss_batch_list = []
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_batch = y_batch.unsqueeze(1)

                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                train_loss_batch_list.append(loss.item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

                y_pred = y_pred.cpu()
                y_batch = y_batch.cpu()
                train_acc_batch = accuracy_score(y_batch, (y_pred > 0.5))
                train_acc_batch_list.append(train_acc_batch)
                train_recall_batch = recall_score(y_batch, (y_pred > 0.5))
                train_recall_batch_list.append(train_recall_batch)
                bar.set_postfix(
                    loss=float(loss),
                    acc=train_acc_batch
                )

        train_acc = np.mean(train_acc_batch_list)
        train_recall = np.mean(train_recall_batch_list)
        train_loss = np.mean(train_loss_batch_list)

        res['train_acc'].append(train_acc)
        res['train_recall'].append(train_recall)
        res['train_loss'].append(train_loss)
        # evaluate accuracy at end of each epoch
        model.eval()
        X_test = X_test.to(device)
        y_pred = model(X_test)
        y_pred = y_pred.cpu()

        test_acc = accuracy_score(y_test, (y_pred > 0.5))
        test_recall = recall_score(y_test, (y_pred > 0.5))
        res['test_acc'].append(test_acc)
        res['test_recall'].append(test_recall)

        test_f1_score = f1_score(y_test, (y_pred > 0.5))

        if test_f1_score > best_f1:
            best_f1 = test_f1_score
            best_weights = copy.deepcopy(model.state_dict())
            train_acc_best = train_acc
            train_recall_best = train_recall
            test_acc_best = test_acc
            test_recall_best = test_recall

    fit_time = time.time() - st
    # save best model
    torch.save(best_weights, os.path.join(
        BASE_DIR,
        'models',
        f'nn_model_{dr_algo}_{cl_algo}.pkl'
    ))

    best = {
        'train_acc': train_acc_best,
        'train_recall': train_recall_best,
        'test_acc': test_acc_best,
        'test_recall': test_recall_best,
        'fit_time': fit_time,
    }

    res['best'] = best

    return res


def nn_run_dr_cl(dr_algo: str = 'No_Dim_Reduction', cl_algo: str = 'No_Clustering'):
    assert dr_algo in ['No_Dim_Reduction', 'PCA', 'ICA', 'RCA', 'ISO']
    assert cl_algo in ['No_Clustering', 'EM', 'KMeans']

    X, y = preprocess_credit_card_data()

    if dr_algo != 'No_Dim_Reduction':
        if dr_algo == 'PCA':
            # dr = PCA(n_components=12, random_state=1)
            dr = PCA(n_components=6, random_state=1)
        elif dr_algo == 'ICA':
            # dr = FastICA(n_components=6, random_state=10)
            dr = FastICA(n_components=3, random_state=10)
        elif dr_algo == 'RCA':
            dr = SparseRandomProjection(n_components=16, random_state=1)
        else:  # dr_algo == 'ISO'
            # dr = Isomap(n_components=9)
            dr = Isomap(n_components=6)
        X = dr.fit_transform(X)

    if cl_algo != 'No_Clustering':
        if cl_algo == 'EM':
            clusterer = GaussianMixture(n_components=2, covariance_type='diag',
                                        n_init=1, warm_start=True, random_state=100)
        else:  # cl_algo == 'KMeans'
            clusterer = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=100)

        clusterer.fit(X)
        cluster_labels = clusterer.predict(X)
        X = np.append(X, cluster_labels.reshape(-1, 1), axis=1)

    X_train, X_test, y_train, y_test = normal_train_test_split(X, y, test_size_=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Then best parameters
    res = model_train(
        X_train, y_train, X_test, y_test,
        model_params=[256, 128, 32],
        n_epochs=200,
        lr=5e-5,
        batch_size=128,
        dr_algo=dr_algo,
        cl_algo=cl_algo
    )

    res['dr_algo'] = dr_algo
    res['cl_algo'] = cl_algo

    return res


def plot_nn_res(input_res: Dict) -> None:
    epoch_num = [i+1 for i in range(len(input_res['train_acc']))]
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].plot(epoch_num, input_res['train_acc'], label='train_acc', color='b')
    ax[0].plot(epoch_num, input_res['test_acc'], label='test_acc', color='r')
    ax[0].set_ylabel('accuracy', fontsize=10)
    ax[0].set_xlabel(f'epoch_num', fontsize=10)
    ax[0].set_title('accuracy vs epoch_num', fontsize=10)
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(epoch_num, input_res['train_recall'], label='train_recall', color='b')
    ax[1].plot(epoch_num, input_res['test_recall'], label='test_recall', color='r')
    ax[1].set_ylabel('recall', fontsize=10)
    ax[1].set_xlabel(f'epoch_num', fontsize=10)
    ax[1].set_title('recall vs epoch_num', fontsize=10)
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(epoch_num, input_res['train_loss'], label='train_loss', color='b')
    ax[2].set_ylabel('train_loss', fontsize=10)
    ax[2].set_xlabel(f'epoch_num', fontsize=10)
    ax[2].set_title('train_loss vs epoch_num', fontsize=10)
    ax[2].legend()
    ax[2].grid()

    fig.suptitle(f"Neural Network Run on Dataset 1: Dimension Reduction = {input_res['dr_algo']}, "
                 f"clustering = {input_res['cl_algo']}")
    fig.savefig(os.path.join(BASE_DIR, 'plots', 'nn',
                             f"nn_{input_res['dr_algo']}_{input_res['cl_algo']}.png"),
                bbox_inches='tight', transparent=False)
    plt.close()


def run_nn(save_res=True):
    all_res = {}
    for dr_algo_ in ['No_Dim_Reduction', 'PCA', 'ICA', 'RCA', 'ISO']:
        for cl_algo_ in ['No_Clustering', 'EM', 'KMeans']:
            print(f'Running dimension reduction = {dr_algo_}, and clustering = {cl_algo_}')
            tmp_res = nn_run_dr_cl(dr_algo=dr_algo_, cl_algo=cl_algo_)
            plot_nn_res(tmp_res)
            all_res[f'{dr_algo_}_{cl_algo_}'] = tmp_res
    if save_res:
        with open(os.path.join(BASE_DIR, 'results', 'all_nn_res.pkl'), 'wb') as file:
            pickle.dump(all_res, file)

    with open(os.path.join(BASE_DIR, 'results', 'all_nn_res.pkl'), 'rb') as file:
        all_res = pickle.load(file)

    out = {
        'run_configs': [],
        'train_acc': [],
        'train_recall': [],
        'test_acc': [],
        'test_recall': [],
        'fit_time': [],
    }

    for run_config in all_res.keys():
        out['run_configs'].append(run_config)
        out['train_acc'].append(all_res[run_config]['best']['train_acc'])
        out['train_recall'].append(all_res[run_config]['best']['train_recall'])
        out['test_acc'].append(all_res[run_config]['best']['test_acc'])
        out['test_recall'].append(all_res[run_config]['best']['test_recall'])
        out['fit_time'].append(all_res[run_config]['best']['fit_time'])
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(BASE_DIR, "results", "nn_results.csv"))

    return all_res
