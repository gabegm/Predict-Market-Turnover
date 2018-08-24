import pandas as pd
from argparse import ArgumentParser
from difflib import SequenceMatcher


def get_features(df):
    """[summary]
    
    Args:
        df ([type]): [description]
    
    Returns:
        [type]: [description]
    """

    # add day of week
    df['weekday'] = df['eventbegin'].dt.weekday_name

    # add period of day
    df = df.assign(
        period=pd.cut(
            x=df['eventbegin'].dt.hour,
            bins=[-1, 12, 17, 24],
            labels=['Morning', 'Afternoon', 'Evening']
        )
    )

    # split datetime
    new_dates, new_times = zip(*[(d.date(), d.time()) for d in df['eventbegin']])
    df = df.assign(date=new_dates, time=new_times).drop(labels=['eventbegin'], axis=1)

    # split participants
    df['participant_a'], df['participant_b'] = df['name_de'].str.strip().str.split(pat=' - ', n=1).str
    df = df.drop(labels=['name_de'], axis=1)

    return df


class SequenceCorrector:
    """Iterator for looping over a sequence backwards.
    
    Returns:
        [type]: [description]
    """

    def __init__(self, s):
        self.s = s
        self.s_unique = s.unique()

    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def correct_duplicates(self):
        for a in self.s_unique:
            for b in self.s_unique:
                result = self.similar(a, b)
                if result > .6 and result < 1.0:
                    print('to_replace =', a, ':', b, 'similarity =', result)
                    self.s = self.s.replace(to_replace={a:b})

        return self.s


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

    df = pd.read_csv(args.filename, parse_dates=['eventbegin'])

    df = get_features(df)

    sc = SequenceCorrector(df['participant_a'])
    df['participant_a'] = sc.correct_duplicates()

    sc = SequenceCorrector(df['participant_b'])
    df['participant_b'] = sc.correct_duplicates()

    df.to_parquet(fname=args.filename.split('.')[0] + '.parquet', engine='pyarrow')