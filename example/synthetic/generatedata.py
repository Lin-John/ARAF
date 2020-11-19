import numpy as np
import pandas as pd


def random_one_zero(size, prob):
    return np.where(np.random.uniform(size=size)<prob, 1, 0)


def generate_multiclass_ub(size=1000):
    probs = [0.3] + [0.5] * 99
    sparse_features = [str(i+1) for i in range(len(probs))]
    dense_features = []
    target = ['y']
    data = {}
    for i, prob in enumerate(probs):
        data[str(i+1)] = random_one_zero(size, prob)
    y = np.zeros(size)
    y = np.where((data['1']==1) & ((data['2']==0) | (data['3']==0)), 1, y)
    y = np.where((data['1']==1) & (data['2']==1) & (data['3']==1), 2, y)
    y = np.where(np.random.uniform(size=size)<=0.05, np.random.randint(3, size=size), y)
    data['y'] = y.astype('int')
    return pd.DataFrame(data), dense_features, sparse_features, target, [], "multiclass"


def generate_multiclass_rdd(size=1000):
    probs = [0.3] + [0.5] * 97 + [1] * 2
    sparse_features = [str(i+1) for i in range(len(probs))]
    dense_features = []
    target = ['y']
    data = {}
    for i, prob in enumerate(probs):
        data[str(i+1)] = random_one_zero(size, prob)
    y = np.zeros(size)
    y = np.where((data['1']==1) & ((data['2']==0) | (data['3']==0)), 1, y)
    y = np.where((data['1']==1) & (data['2']==1) & (data['3']==1), 2, y)
    y = np.where(np.random.uniform(size=size)<=0.05, np.random.randint(3, size=size), y)
    data['y'] = y.astype('int')
    return pd.DataFrame(data), dense_features, sparse_features, target, [], "multiclass"