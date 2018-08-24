import random as rand
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from functions import encode_labels_test


def split_dataset(df):
    """cross-validation testing

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """

    split = rand.uniform(0.60, 0.80)

    train_size = int(len(df) * split)

    train, test = df[0:train_size], df[train_size:len(df)]

    return train, test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to file",
        metavar="FILE",
        dest="filename"
    )

    args = parser.parse_args()

    df = pd.read_parquet(path=args.filename, engine='pyarrow')

    df = encode_labels_test(df)

    features = np.delete(
        arr=df.columns.values,
        obj=np.where(
            df.columns.values == 'turnover'
        )
    )

    train, test = split_dataset(df)

    # values of features
    X = np.array(train[features].values)

    # target values
    y = list(train['turnover'])

    dtr = DecisionTreeRegressor()

    # Set the parameters by cross-validation
    parameters = [
        {
            'max_features': ['sqrt', 'log2', None],
            'max_depth': range(2, 1000),
        }
    ]

    clf = GridSearchCV(dtr, parameters)
    clf.fit(X, y)
    
    # make predictions
    pred = clf.predict(test[features])

    results = pd.DataFrame(
        data={
            'original': test['turnover'],
            'prediction': pred
        },
        index=test.index
    )

    # summarize the fit of the model
    r2 = r2_score(test['turnover'].values, pred)
    rms = mean_squared_error(test['turnover'].values, pred)
    metrics = pd.DataFrame(
        data={
            'r2_score': r2, 'mean_squared_error': rms
        },
        index=[0]
    )

    results.to_csv('data/metrics/results.csv')
    metrics.to_csv('data/metrics/metrics.csv')
