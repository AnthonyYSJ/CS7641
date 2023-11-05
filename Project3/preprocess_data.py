import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from settings import BASE_DIR


def preprocess_credit_card_data(normalize=True, verbose=False):
    data = pd.read_csv(os.path.join(BASE_DIR, 'data', "BankChurners.csv"))

    # drop the irrelevant columns
    data.drop(columns=[col for col in data.columns if col.startswith("Naive_Bayes")], inplace=True)
    data.drop(columns=["CLIENTNUM"], inplace=True)

    # first let's see if there's any null variables
    data.isnull().any().any()

    if verbose:
        print(data.info)

    # let's deal with non-numeric columns

    non_numeric_cols = [x for x in data.dtypes.index if (data.dtypes[x] != "int64") & (data.dtypes[x] != "float64")]

    for col in non_numeric_cols:
        if verbose:
            print(f'{col}: {data[col].unique().tolist()}')

    data['Attrition_Flag'] = data['Attrition_Flag'].map({"Existing Customer": 1, "Attrited Customer": 0})
    data['Gender'] = data['Gender'].map({"M": 1, "F": 0})
    data['Education_Level'] = data['Education_Level'].map(
        {
            'Unknown': 0,
            'Uneducated': 1,
            'High School': 2,
            'College': 3,
            'Graduate': 4,
            'Post-Graduate': 5,
            'Doctorate': 6,
        }
    )
    data['Marital_Status'] = data['Marital_Status'].map(
        {
            'Unknown': 0,
            'Single': 1,
            'Married': 2,
            'Divorced': 3,
        }
    )
    data['Income_Category'] = data['Income_Category'].map(
        {
            'Unknown': 0,
            'Less than $40K': 1,
            '$40K - $60K': 2,
            '$60K - $80K': 3,
            '$80K - $120K': 4,
            '$120K +': 5,
        }
    )
    data['Card_Category'] = data['Card_Category'].map(
        {
            'Blue': 0,
            'Silver': 1,
            'Gold': 2,
            'Platinum': 3,
        }
    )

    y = data.loc[:, "Attrition_Flag"].values
    y = y.reshape(-1, 1)
    X = data.loc[:, [col for col in data.columns if col != "Attrition_Flag"]].values

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y


def preprocess_customer_segmentation_data(normalize=True, verbose=False):
    train_data = pd.read_csv(os.path.join(BASE_DIR, 'data', "customer_segmentation", "Train.csv"))
    test_data = pd.read_csv(os.path.join(BASE_DIR, 'data', "customer_segmentation", "Test.csv"))

    data = pd.concat([train_data, test_data])
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.drop(columns=['ID'], inplace=True)

    # first let's see if there's any null variables
    assert not data.isnull().any().any()

    if verbose:
        print(data.info)

    # let's deal with non-numeric columns
    non_numeric_cols = [x for x in data.dtypes.index if (data.dtypes[x] != "int64") & (data.dtypes[x] != "float64")]

    for col in non_numeric_cols:
        if verbose:
            print(f'{col}: {data[col].unique().tolist()}')

    data['Gender'] = data['Gender'].map({"Male": 1, "Female": 0})
    data['Ever_Married'] = data['Ever_Married'].map({"Yes": 1, "No": 0})
    data['Graduated'] = data['Graduated'].map({"Yes": 1, "No": 0})
    data['Spending_Score'] = data['Spending_Score'].map({"Low": 0, "Average": 1, "High": 2})

    for col in ['Profession', 'Var_1', 'Segmentation']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    y = data.loc[:, 'Segmentation'].values
    y = y.reshape(-1, 1)
    X = data.loc[:, [col for col in data.columns if col != 'Segmentation']].values

    if normalize:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

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
