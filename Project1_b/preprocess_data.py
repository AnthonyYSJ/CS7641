import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from settings import BASE_DIR


def preprocess_credit_card_data(verbose=False, plot=False):
    data = pd.read_csv(os.path.join(BASE_DIR, 'data', "creditcard.csv"))

    # first let's see if there's any null variables
    data.isnull().any().any()

    # let's deal with non-numeric columns
    non_numeric_cols = [x for x in data.dtypes.index if (data.dtypes[x] != "int64") & (data.dtypes[x] != "float64")]

    # show that all columns are numeric!
    assert not bool(non_numeric_cols)

    # only two columns are not scaled, let's take a look at the distributions first
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        amount_val = data['Amount'].values
        time_val = data['Time'].values

        sns.distplot(amount_val, ax=ax[0])
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

        sns.distplot(time_val, ax=ax[1])
        ax[1].set_title('Distribution of Transaction Time', fontsize=14)

        plt.savefig(os.path.join(BASE_DIR, 'plots', 'data_fields_distribution.png'))
        plt.close()

    # looking at distribution, we need to use some scaler, and we use RobustScaler because transaction
    # amount have some very clear outliers
    scaler = RobustScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

    # Then we address the issue of class imbalance
    if verbose:
        print(data.info)
        fraud_count = data['Class'][data['Class'] == 1].count()
        non_fraud_count = data.shape[0] - fraud_count
        print(f'Number of fraud cases = {fraud_count}')
        print(f'Number of non-fraud cases = {non_fraud_count}')
        print(f'Fraud percentage = {round(fraud_count / (fraud_count + non_fraud_count) * 100, 2)}%')
        print(f'None-fraud percentage = {round(non_fraud_count / (fraud_count + non_fraud_count) * 100, 2)}%')

    y = data.loc[:, "Class"].values
    y = y.reshape(-1, 1)
    X = data.loc[:, [col for col in data.columns if col != "Class"]].values

    return X, y


def normal_train_test_split(X: DataFrame, y: DataFrame, test_size_: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_,
        random_state=44,
        shuffle=True
    )

    y_train = y_train.reshape(-1, )
    y_test = y_test.reshape(-1, )

    return X_train, X_test, y_train, y_test


def smote_train_test_split(X: DataFrame, y: DataFrame, test_size_: float = 0.2, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_,
        random_state=44,
        shuffle=True
    )

    y_test = y_test.reshape(-1, )

    oversample_sampler = SMOTE(sampling_strategy='minority', random_state=0)
    X_train, y_train = oversample_sampler.fit_resample(X_train, y_train)

    if verbose:
        print(collections.Counter(y_train))
        print(collections.Counter(y_test))

    return X_train, X_test, y_train, y_test
