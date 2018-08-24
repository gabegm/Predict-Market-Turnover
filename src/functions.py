import pandas as pd


def encode_labels_test(df):
    """[summary]
    
    Args:
        df ([type]): [description]
    
    Returns:
        [type]: [description]
    """


    features = [
        'channel',
        'market',
        'weekday',
        'period',
        'participant_a',
        'participant_b'
    ]

    df = pd.get_dummies(
        data=df,
        prefix='is_',
        columns=features
    ).drop(
        labels=['programevent_id', 'date', 'time'],
        axis=1
    )

    return df


def encode_labels(df1, df2):
    """Convert categorical variables into dummy/indicator variables
    
    Args:
        df1 ([type]): [description]
        df2 ([type]): [description]
    
    Returns:
        [type]: [description]
    """


    features = [
        'channel',
        'market',
        'weekday',
        'period',
        'participant_a',
        'participant_b'
    ]

    df1['label'] = 'train'
    df2['label'] = 'predict'

    # Concat
    concat_df = pd.concat([df1 , df2])

    df = pd.get_dummies(
        data=concat_df,
        prefix='is_',
        columns=features
    ).drop(
        labels=['programevent_id', 'date', 'time'],
        axis=1
    )

    df = df.loc[:, ~df.columns.duplicated()]

    # Split your data
    df1 = df[df['label'] == 'train']
    df2 = df[df['label'] == 'predict']

    # Drop your labels
    df1 = df1.drop(labels='label', axis=1)
    df2 = df2.drop(labels='label', axis=1)

    return df1, df2