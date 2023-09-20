import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt

from preprocess_data import preprocess_credit_card_data, normal_train_test_split
from settings import BASE_DIR


class Net_relu(nn.Module):
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


def model_train(X_train, y_train, X_val, y_val, X_test, y_test,
                model_params=None, n_epochs=100, lr=1e-5, batch_size=64):
    if model_params is None:
        model_params = [512, 64, 16]
    # loss function and optimizer
    model = Net_relu(input_dim=X_train.shape[1], hidden_layers=model_params)
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf  # init to negative infinity
    best_weights = None

    st = time.time()
    train_results = []
    val_results = []
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                train_acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(train_acc)
                )

        train_results.append(float(train_acc))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        # with torch.no_grad():
        #     y_pred = model(X_val)
        val_acc = (y_pred.round() == y_val).float().mean()
        val_acc = float(val_acc)
        val_results.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    duration = time.time() - st
    # save best model
    torch.save(model.state_dict(), os.path.join(
        BASE_DIR,
        'models',
        'nn_models',
        f'model_{model_params}_batch_size_{batch_size}_lr_{lr}.pkl'
    ))

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    model.eval()
    y_pred = model(X_test)
    test_acc = (y_pred.round() == y_test).float().mean()
    test_acc = float(test_acc)

    best_performance = {
        'train': float(train_acc),
        'validation': best_acc,
        'test': test_acc,
        'duration': duration
    }

    epoch_performance = {
        'train_performance': train_results,
        'validation_performance': val_results
    }

    model_specs = {
        'n_epochs': n_epochs,
        'model_params': model_params,
        'lr': lr,
        'batch_size': batch_size,
    }

    return best_performance, epoch_performance, model_specs


def plot_epoch_performance(epoch_performance, model_specs):
    x_axis_len = len(epoch_performance['train_performance'])
    plt.plot(
        list(range(x_axis_len)),
        epoch_performance['train_performance'],
        label="train"
    )
    plt.plot(
        list(range(x_axis_len)),
        epoch_performance['validation_performance'],
        label="validation"
    )
    plt.legend()
    plt.title(f'Accuracy: {str(model_specs)}')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)

    plot_name = ('nn_acc_epoch_performance' + '_model_params_' + str(model_specs['model_params']) + '_n_epochs_' +
                 str(model_specs['n_epochs']) + '_lr_' + str(model_specs['lr']) + '_batch_size_' +
                 str(model_specs['batch_size']))
    plt.savefig(
        os.path.join(BASE_DIR, 'plots', f'{plot_name}.png'),
        dpi=300
    )
    plt.close()


def plot_best_epoch_performance(epoch_performance, model_specs):
    x_axis_len = len(epoch_performance['train_performance'])
    plt.plot(
        list(range(x_axis_len)),
        epoch_performance['train_performance'],
        label="train"
    )
    plt.plot(
        list(range(x_axis_len)),
        epoch_performance['validation_performance'],
        label="validation"
    )
    plt.legend()
    plt.title(f'Accuracy: {str(model_specs)}')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)

    plot_name = ('nn_best_epoch_performance' + '_model_params_' + str(model_specs['model_params']) + '_n_epochs_' +
                 str(model_specs['n_epochs']) + '_lr_' + str(model_specs['lr']) + '_batch_size_' +
                 str(model_specs['batch_size']))
    plt.savefig(
        os.path.join(BASE_DIR, 'plots', f'{plot_name}.png'),
        dpi=300
    )
    plt.close()


def plot_param_performance(result_dict):
    param_type = list(set(result_dict['param_type']))[0]
    plt.plot(
        result_dict['param_val'],
        result_dict['train'],
        label="train"
    )
    plt.plot(
        result_dict['param_val'],
        result_dict['validation'],
        label="validation"
    )
    plt.plot(
        result_dict['param_val'],
        result_dict['test'],
        label="test"
    )
    plt.legend()
    plt.ticklabel_format(axis='both', style='sci')
    plt.title(f'Accuracy vs {str(param_type)}')
    plt.xlabel(f'{param_type}', fontsize=10)
    plt.ylabel('accuracy', fontsize=10)
    if param_type == 'lr':
        plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.xticks(fontsize=8)

    plt.savefig(
        os.path.join(BASE_DIR, 'plots', f'nn_acc_best_performance_vs_{param_type}.png'),
        dpi=300
    )
    plt.close()


def tune_model_size(data, model_params_list, n_epochs=100, lr=1e-4, batch_size=128):
    results = {
        'model_params': [],
        'train': [],
        'validation': [],
        'test': [],
        'duration': [],
    }

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    for model_params in model_params_list:
        print(model_params)
        results['model_params'].append(str(model_params))
        best_performance, epoch_performance, model_specs = model_train(
            X_train, y_train, X_val, y_val, X_test, y_test,
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
        'train': [],
        'validation': [],
        'test': [],
        'duration': [],
    }

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    for param_val in param_range:
        print(param_val)
        results['param_type'].append(param_type)
        results['param_val'].append(param_val)

        if param_type == 'n_epoch':
            best_performance, epoch_performance, model_specs = model_train(
                X_train, y_train, X_val, y_val, X_test, y_test,
                model_params=[512, 128, 16],
                n_epochs=param_val,
                lr=1e-4,
                batch_size=64,
            )
        elif param_type == 'lr':
            best_performance, epoch_performance, model_specs = model_train(
                X_train, y_train, X_val, y_val, X_test, y_test,
                model_params=[512, 128, 16],
                n_epochs=200,
                lr=param_val,
                batch_size=64,
            )
        else:
            # param_type == 'batch_size':
            best_performance, epoch_performance, model_specs = model_train(
                X_train, y_train, X_val, y_val, X_test, y_test,
                model_params=[512, 128, 16],
                n_epochs=200,
                lr=1e-4,
                batch_size=param_val,
            )

        for key in best_performance:
            results[key].append(best_performance[key])

    plot_param_performance(results)
    results = pd.DataFrame(results)
    csv_name = f'nn_model_hyper_parameter_tune_{param_type}.csv'
    results.to_csv(os.path.join(BASE_DIR, 'results', csv_name), index=False)
    return results


def run_neural_network():
    X, y = preprocess_credit_card_data()
    X_train, X_test, y_train, y_test = normal_train_test_split(X, y, test_size_=0.1)
    X_train, X_val, y_train, y_val = normal_train_test_split(X_train, y_train, test_size_=0.2)

    data = {'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.float32),
            'X_val': torch.tensor(X_val, dtype=torch.float32),
            'y_val': torch.tensor(y_val, dtype=torch.float32),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.float32)}

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # torch.backends.cudnn.benchmark = True

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

    tune_model_size(data, model_params_list, n_epochs=100, lr=1e-4, batch_size=128)
    tune_model_size(data, model_params_list, n_epochs=200, lr=1e-4, batch_size=64)
    tune_model_size(data, model_params_list, n_epochs=200, lr=1e-5, batch_size=256)

    tune_other_params(data, 'lr', np.arange(1e-5, 2e-4, 1e-5))
    tune_other_params(data, 'batch_size', [32, 64, 128, 256, 512])
    tune_other_params(data, 'n_epoch', np.arange(100, 1100, 100))

    # Then best parameters
    best_performance, epoch_performance, model_specs = model_train(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_params=[512, 128, 16],
        n_epochs=300,
        lr=1.2e-4,
        batch_size=64,
    )
    plot_best_epoch_performance(epoch_performance, model_specs)
