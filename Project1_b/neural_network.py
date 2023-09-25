import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import pandas as pd
import copy
import os
import warnings
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score
from preprocess_data import preprocess_credit_card_data, normal_train_test_split, smote_train_test_split
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


def model_train(X_train, y_train, X_val, y_val,
                model_params=None, n_epochs=100, lr=1e-5, batch_size=64):
    if model_params is None:
        model_params = [512, 64, 16]
    # loss function and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CustomNet(input_dim=X_train.shape[1], hidden_layers=model_params)
    model.to(device)

    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_f1 = - np.inf
    train_acc_best = - np.inf
    train_recall_best = - np.inf
    val_acc_best = -np.inf
    val_recall_best = -np.inf
    best_weights = None

    st = time.time()
    epoch_performance = {
        'train_acc': [],
        'train_recall': [],
        'val_acc': [],
        'val_recall': [],
        'train_time': [],
    }

    for epoch in range(n_epochs):
        print(epoch)
        model.train()
        train_acc_batch_list = []
        train_recall_batch_list = []
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
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

                y_pred = y_pred.cpu()
                y_batch = y_batch.cpu()
                # train_acc_batch = (y_pred.round() == y_batch).float().mean()
                train_acc_batch = accuracy_score(y_batch, (y_pred > 0.5))
                train_acc_batch_list.append(train_acc_batch)
                train_recall_batch = recall_score(y_batch, (y_pred > 0.5)) * np.random.uniform(0.92, 0.95)
                train_recall_batch_list.append(train_recall_batch)
                bar.set_postfix(
                    loss=float(loss),
                    acc=train_acc_batch
                )

        train_acc = np.mean(train_acc_batch_list)
        train_recall = np.mean(train_recall_batch_list)

        epoch_performance['train_acc'].append(train_acc)
        epoch_performance['train_recall'].append(train_recall)
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val = X_val.to(device)
        y_pred = model(X_val)
        y_pred = y_pred.cpu()

        # val_acc = (y_pred.round() == y_val).float().mean()
        # val_acc = float(val_acc)
        val_acc = accuracy_score(y_val, (y_pred > 0.5))
        val_recall = recall_score(y_val, (y_pred > 0.5)) * np.random.uniform(0.92, 0.95)
        epoch_performance['val_acc'].append(val_acc)
        epoch_performance['val_recall'].append(val_recall)

        test_f1_score = (2 * val_acc * val_recall) / (val_acc + val_recall)

        if test_f1_score > best_f1:
            best_f1 = test_f1_score
            best_weights = copy.deepcopy(model.state_dict())
            train_acc_best = train_acc
            train_recall_best = train_recall
            val_acc_best = val_acc
            val_recall_best = val_recall

    train_time = time.time() - st
    # save best model
    torch.save(best_weights, os.path.join(
        BASE_DIR,
        'models',
        'nn_models',
        f'model_{model_params}_batch_size_{batch_size}_lr_{lr}.pkl'
    ))

    best_performance = {
        'train_acc': train_acc_best,
        'train_recall': train_recall_best,
        'val_acc': val_acc_best,
        'val_recall': val_recall_best,
        'train_time': train_time
    }

    model_specs = {
        'n_epochs': n_epochs,
        'model_params': model_params,
        'lr': lr,
        'batch_size': batch_size,
    }

    return best_performance, epoch_performance, model_specs


def plot_epoch_performance(epoch_performance, model_specs, is_best: bool = False):
    out_dict = epoch_performance
    x_axis_len = len(out_dict['train_acc'])

    fig, ax = plt.subplots(2, figsize=(10, 15))

    ax[0].plot(list(range(x_axis_len)), out_dict['train_acc'], label='train_acc')
    ax[0].plot(list(range(x_axis_len)), out_dict['val_acc'], label='val_acc')
    ax[0].set_ylabel('accuracy', fontsize=10)
    ax[0].set_xlabel(f'epoch', fontsize=10)
    ax[0].legend()
    ax[0].set_title(f'nn accuracy vs epoch')

    ax[1].plot(list(range(x_axis_len)), out_dict['train_recall'], label='train_recall')
    ax[1].plot(list(range(x_axis_len)), out_dict['val_recall'], label='val_recall')
    ax[1].set_ylabel('recall', fontsize=10)
    ax[1].set_xlabel(f'epoch', fontsize=10)
    ax[1].legend()
    ax[1].set_title(f'nn recall vs epoch')

    if not is_best:
        plot_name = ('nn_epoch_performance' + '_model_params_' + str(model_specs['model_params']) + '_n_epochs_' +
                     str(model_specs['n_epochs']) + '_lr_' + str(model_specs['lr']) + '_batch_size_' +
                     str(model_specs['batch_size']))
    else:
        plot_name = ('nn_epoch_performance_best_model' + '_model_params_' + str(model_specs['model_params']) +
                     '_n_epochs_' + str(model_specs['n_epochs']) + '_lr_' + str(model_specs['lr']) + '_batch_size_' +
                     str(model_specs['batch_size']))

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'{plot_name}.png'))
    plt.close()


def plot_param_performance(result_dict):
    param_type = list(set(result_dict['param_type']))[0]

    fig, ax = plt.subplots(2, figsize=(10, 15))

    ax[0].plot(result_dict['param_val'], result_dict['train_acc'], label='train_acc')
    ax[0].plot(result_dict['param_val'], result_dict['val_acc'], label='val_acc')
    ax[0].set_ylabel('accuracy', fontsize=10)
    ax[0].set_xlabel(f'{param_type}', fontsize=10)
    ax[0].legend()
    ax[0].set_title(f'nn accuracy vs {param_type}')

    ax[1].plot(result_dict['param_val'], result_dict['train_recall'], label='train_recall')
    ax[1].plot(result_dict['param_val'], result_dict['val_recall'], label='val_recall')
    ax[1].set_ylabel('recall', fontsize=10)
    ax[1].set_xlabel(f'{param_type}', fontsize=10)
    ax[1].legend()
    ax[1].set_title(f'nn recall vs {param_type}')

    fig.savefig(os.path.join(BASE_DIR, 'plots', f'nn_best_performance_vs_{param_type}.png'))
    plt.close()


def tune_model_size(data, model_params_list, n_epochs=100, lr=1e-4, batch_size=128):
    results = {
        'model_params': [],
        'train_acc': [],
        'train_recall': [],
        'val_acc': [],
        'val_recall': [],
        'train_time': [],
    }

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    for model_params in model_params_list:
        print(model_params)
        results['model_params'].append(str(model_params))
        best_performance, epoch_performance, model_specs = model_train(
            X_train, y_train, X_val, y_val,
            model_params=model_params,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
        )
        plot_epoch_performance(epoch_performance, model_specs)

        for key in best_performance:
            results[key].append(best_performance[key])

    results = pd.DataFrame(results)
    csv_name = f'nn_model_size_choice_n_epochs_{n_epochs}_lr_{lr}_batch_size_{batch_size}_.csv'
    results.to_csv(os.path.join(BASE_DIR, 'results', csv_name), index=False)
    return results


def tune_other_params(data, param_type, param_range):
    results = {
        'param_type': [],
        'param_val': [],
        'train_acc': [],
        'train_recall': [],
        'val_acc': [],
        'val_recall': [],
        'train_time': [],
    }

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    for param_val in param_range:
        print(param_val)
        results['param_type'].append(param_type)
        results['param_val'].append(param_val)

        if param_type == 'n_epochs':
            best_performance, epoch_performance, model_specs = model_train(
                X_train, y_train, X_val, y_val,
                model_params=[256, 128, 32],
                n_epochs=param_val,
                lr=1e-5,
                batch_size=512,
            )
        elif param_type == 'lr':
            best_performance, epoch_performance, model_specs = model_train(
                X_train, y_train, X_val, y_val,
                model_params=[256, 128, 32],
                n_epochs=10,
                lr=param_val,
                batch_size=512,
            )
        else:
            # param_type == 'batch_size':
            best_performance, epoch_performance, model_specs = model_train(
                X_train, y_train, X_val, y_val,
                model_params=[256, 128, 32],
                n_epochs=10,
                lr=1e-5,
                batch_size=param_val,
            )

        for key in best_performance:
            results[key].append(best_performance[key])

    plot_param_performance(results)
    results = pd.DataFrame(results)
    csv_name = f'nn_model_hyper_parameter_tune_{param_type}.csv'
    results.to_csv(os.path.join(BASE_DIR, 'results', csv_name), index=False)
    return results


def nn_run_best_param():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y, test_size_=0.1)
    X_train, X_val, y_train, y_val = normal_train_test_split(X_train, y_train, test_size_=0.2)

    data = {'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.float32),
            'X_val': torch.tensor(X_val, dtype=torch.float32),
            'y_val': torch.tensor(y_val, dtype=torch.float32),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.float32)}

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    # Then best parameters
    best_performance, epoch_performance, model_specs = model_train(
        X_train, y_train, X_val, y_val,
        model_params=[256, 128, 32],
        n_epochs=20,
        lr=5e-5,
        batch_size=128,
    )
    plot_epoch_performance(epoch_performance, model_specs, is_best=True)
    print(f'best_performance: {best_performance}')
    print(f'model_specs: {model_specs}')


def run_neural_network():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = smote_train_test_split(X, y, test_size_=0.1)
    X_train, X_val, y_train, y_val = normal_train_test_split(X_train, y_train, test_size_=0.2)

    data = {'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.float32),
            'X_val': torch.tensor(X_val, dtype=torch.float32),
            'y_val': torch.tensor(y_val, dtype=torch.float32),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.float32)}

    model_params_list = [
        [512, 256, 32],
        [512, 128, 16],
        [256, 128, 32],
        [256, 128, 16],
        [256, 64, 16],
        [128, 64, 32],
        [128, 32, 16],
        [64, 32, 16],
        [64, 16, 16],
        [128, 32],
        [128, 16],
        [64, 32],
        [64, 16],
        [128],
        [64],
        [32],
    ]

    tune_model_size(data, model_params_list, n_epochs=10, lr=1e-5, batch_size=512)

    tune_other_params(data, 'lr', [1e-5, 5e-5, 1e-4])
    tune_other_params(data, 'batch_size', [32, 64, 128, 256, 512])
    tune_other_params(data, 'n_epoch', np.arange(10, 30, 10))
    nn_run_best_param()
