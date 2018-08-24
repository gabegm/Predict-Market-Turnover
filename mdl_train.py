import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from functions import encode_labels

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        nargs="+",
        required=True,
        help="Path to file",
        metavar="FILE",
        dest="filename"
    )

    args = parser.parse_args()

    df1, df2 = [pd.read_parquet(path=file, engine='pyarrow') for file in args.filename]

    df1, df2 = encode_labels(df1, df2)

    # save predict data set
    df2.to_parquet(fname=args.filename[1], engine='pyarrow')

    features = np.delete(
        arr=df1.columns.values,
        obj=np.where((df1.columns.values == 'turnover') | (df1.columns.values == 'label'))
    )

    # values of features
    X = np.array(df1[features].values)

    # target values
    y = list(df1['turnover'])

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
    pred = clf.predict(df1[features])

    s = pickle.dump(clf, open("model/mdl.obj", "wb"))
