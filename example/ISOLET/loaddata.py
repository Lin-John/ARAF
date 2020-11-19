import pandas as pd


def load_data_isolet():
    # load data set: Heart Disease
    cols = ['f' + str(i) for i in range(617)] + ['label']
    dense_features = ['f' + str(i) for i in range(617)]
    sparse_features = []
    target = ['label']
    miss_val = ['?']
    task = 'multiclass'

    E_set = [ord(c) - ord('A') + 1 for c in ['B', 'C', 'D', 'E', 'G', 'P', 'T', 'V', 'Z']]
    df1 = pd.read_csv(r"data/ISOLET/isolet1+2+3+4.data", names=cols)
    df1 = df1[df1['label'].isin(E_set)]
    df2 = pd.read_csv(r"data/ISOLET/isolet5.data", names=cols)
    df2 = df2[df2['label'].isin(E_set)]

    return pd.concat((df1, df2), axis=0), dense_features, sparse_features, target, miss_val, task
