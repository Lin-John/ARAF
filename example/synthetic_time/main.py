import time
import matplotlib.pyplot as plt
from araf import ARAF
from generatedata import generate_syn_random


def test_synthetic_n(generate_data, size, n_freq, sample_sizes):
    """Test the estimated frequencies and running time for different sample sizes."""
    df, dense_features, sparse_features, target, miss_val, task = generate_data(size)
    X, y = df[sparse_features+dense_features], df[target].values.ravel()
    rules_freq = [((0, 1),), ((1, 1),), ((2, 1),), ((0, 1),(1,1))]
    freqs = {rule:[0]*len(sample_sizes) for rule in rules_freq}
    times = []
    for i, sample_size in enumerate(sample_sizes):
        # ARAF
        araf = ARAF(n_freq, 0, sample_size=sample_size)
        time_start = time.time()
        araf.fit(X, y)
        times.append(time.time()-time_start)
        for rule, supp in araf.frequent_set[1].items():
            if rule in freqs:
                freqs[rule][i] = supp/sample_size
    return freqs, times


def test_synthetic_p(generate_data, size, n_freq, n_features, sample_size=5000):
    """Test the running time for different feature sizes."""
    times = []
    for p in n_features:
        df, dense_features, sparse_features, target, miss_val, task = generate_data(size, p)
        X, y = df[sparse_features+dense_features], df[target].values.ravel()
        araf = ARAF(n_freq, 5, sample_size=sample_size)
        time_start = time.time()
        araf.fit(X, y)
        times.append(time.time()-time_start)
    return times


def tostring(item):
    """Convert a rule to string."""
    res = []
    for index, value in item:
        res.append('X{}={}'.format(index+1, value))
    return ','.join(res)


if __name__ == "__main__":
    sample_sizes = range(100, 5100, 100)
    freqs, times = test_synthetic_n(generate_syn_random, 10000, 5, sample_sizes)
    x1, x2, x3, x4 = freqs.keys()
    color = {x1: 'y', x2: 'g', x3: 'c', x4: 'r'}
    freq_true = {((0, 1),): 0.9, ((1, 1),): 0.8, ((2, 1),): 0.7, ((0, 1), (1, 1)): 0.8}

    # plot the estimated frequencies
    for x in freqs:
        plt.plot(sample_sizes, freqs[x], color=color[x], label=tostring(x))
        plt.plot(sample_sizes, [freq_true[x]] * len(sample_sizes), color=color[x], linestyle=':')
    plt.xlabel("number of samples")
    plt.ylabel("estimated frequency")
    plt.axis([0, 5000, 0.5, 1])
    plt.legend()
    # plt.savefig('frequency.png', bbox_inches='tight')
    plt.show()

    # plot the running time of different sample sizes
    plt.plot(sample_sizes, times)
    plt.xlabel("number of samples")
    plt.ylabel("running time (s)")
    plt.show()

    # plot the running time of different feature sizes
    n_features = range(10, 101)
    times = test_synthetic_p(generate_syn_random, 10000, 5, n_features)
    plt.plot(n_features, times)
    plt.xlabel("number of features")
    plt.ylabel("running time (s)")
    plt.show()
