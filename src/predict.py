import pickle
import pandas as pd
import numpy as np
from argparse import ArgumentParser


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

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to model",
        metavar="FILE",
        dest="model"
    )

    args = parser.parse_args()

    df = pd.read_parquet(path=args.filename, engine='pyarrow')

    features = np.delete(
        arr=df.columns.values,
        obj=np.where(df.columns.values == 'turnover')
    )

    mdl = pickle.load(open(args.model, "rb"))

    # make predictions
    pred = mdl.predict(df[features])

    pd.DataFrame(
        data={'turnover': pred}
    ).to_csv('data/predict_results.csv')