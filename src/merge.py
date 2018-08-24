import pandas as pd
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

    args = parser.parse_args()

    df = pd.read_csv(args.filename)

    pred = pd.read_csv('../data/predict_results.csv')

    df['turnover'] = pred['turnover']

    df.to_csv('../data/final.csv')