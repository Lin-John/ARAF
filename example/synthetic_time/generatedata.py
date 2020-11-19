import numpy as np
import pandas as pd


def random_one_zero(size, prob):
    return np.where(np.random.uniform(size=size) < prob, 1, 0)


def generate_syn_random(n=10000, p=10):
    sparse_features = [i + 1 for i in range(p)]
    dense_features = []
    target = ['y']
    data = {i: random_one_zero(n, 0.5) for i in range(4, n + 1)}
    data[1] = random_one_zero(n, 0.9)
    data[2] = np.where(1 == data[1], random_one_zero(n, 75 / 90), 0)
    data[2] = np.where(0 == data[1], random_one_zero(n, 0.5), data[2])
    data[3] = random_one_zero(n, 0.7)
    data['y'] = np.ones(n)
    return pd.DataFrame(data), dense_features, sparse_features, target, [], "binary"
